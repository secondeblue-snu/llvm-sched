// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Single scalar reduction on omp.taskloop.context. The lowering must:
//   1. Emit an implicit __kmpc_taskgroup in the encountering function (since
//      the user did not write nogroup);
//   2. Build a kmp_taskred_input_t descriptor array and call
//      __kmpc_taskred_init, capturing the returned descriptor handle;
//   3. Force nogroup=1 on the inner __kmpc_taskloop call so that the
//      OpenMPIRBuilder does not emit a second taskgroup;
//   4. Inside the outlined task body, call __kmpc_global_thread_num to obtain
//      the executing thread's gtid, then look up the per-task private storage
//      via __kmpc_task_reduction_get_th_data(gtid, redDesc, orig);
//   5. Close the implicit taskgroup with __kmpc_end_taskgroup.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskloop_reduction_single(%x : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
  omp.taskloop.context reduction(@add_i32 %x -> %prv : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        %v = llvm.load %prv : !llvm.ptr -> i32
        %s = llvm.add %v, %iv : i32
        llvm.store %s, %prv : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK: %kmp_taskred_input_t = type { ptr, ptr, i64, ptr, ptr, ptr, i32 }

// Encountering function emits taskgroup + descriptor + taskred_init.
// CHECK-LABEL: define void @taskloop_reduction_single(
// CHECK-SAME:    ptr %[[X:[^,]+]],
// CHECK:         %[[ARR:.+]] = alloca [1 x %kmp_taskred_input_t]
// CHECK:         call void @__kmpc_taskgroup(
// CHECK:         %[[ELEM:.+]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK:         %[[SHAR:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[ELEM]], i32 0, i32 0
// CHECK:         store ptr %[[X]], ptr %[[SHAR]]
// CHECK:         store ptr @__omp_taskloop_taskred_add_i32.red.init
// CHECK:         store ptr @__omp_taskloop_taskred_add_i32.red.comb
// CHECK:         %[[DESC:.+]] = call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 1, ptr %[[ARR]])
// The returned descriptor is stored into the structArg captured by
// __kmpc_omp_task_alloc so the outlined task body can load it back.
// CHECK:         store ptr %[[DESC]], ptr %{{.+}}
// __kmpc_taskloop must be called with nogroup=1 because we already opened
// our own taskgroup above.
// CHECK:         call void @__kmpc_taskloop(ptr {{.+}}, i32 {{.+}}, ptr {{.+}}, i32 1,
// CHECK:         call void @__kmpc_end_taskgroup(

// Outlined task body looks up per-task storage via the runtime, passing the
// reloaded descriptor (not null) as the second argument.
// CHECK-LABEL: define internal void @taskloop_reduction_single..omp_par(
// CHECK:         %[[BODY_DESC:.+]] = load ptr, ptr %gep_.taskred.desc
// CHECK:         %[[BODY_ORIG:.+]] = load ptr, ptr %gep_,
// CHECK:         %[[BODY_GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID]], ptr %[[BODY_DESC]], ptr %[[BODY_ORIG]])
// CHECK:         load i32, ptr %[[PRIV]]
// CHECK:         store i32 %{{.+}}, ptr %[[PRIV]]

// -----

// Multiple reductions: each entry in the descriptor array gets distinct
// init / combiner helpers and the body issues one
// __kmpc_task_reduction_get_th_data per reduction.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

omp.declare_reduction @mul_i64 : i64
init {
^bb0(%arg0: i64):
  %c1 = llvm.mlir.constant(1 : i64) : i64
  omp.yield(%c1 : i64)
}
combiner {
^bb0(%arg0: i64, %arg1: i64):
  %p = llvm.mul %arg0, %arg1 : i64
  omp.yield(%p : i64)
}

llvm.func @taskloop_reduction_multi(%x : !llvm.ptr, %y : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
  omp.taskloop.context reduction(@add_i32 %x -> %a, @mul_i64 %y -> %b : !llvm.ptr, !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        %va = llvm.load %a : !llvm.ptr -> i32
        %vai = llvm.add %va, %iv : i32
        llvm.store %vai, %a : i32, !llvm.ptr
        %vb = llvm.load %b : !llvm.ptr -> i64
        %iv64 = llvm.sext %iv : i32 to i64
        %vbi = llvm.mul %vb, %iv64 : i64
        llvm.store %vbi, %b : i64, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @taskloop_reduction_multi(
// CHECK:         %[[ARR2:.+]] = alloca [2 x %kmp_taskred_input_t]
// CHECK:         call void @__kmpc_taskgroup(
// CHECK:         store i64 4
// CHECK:         store ptr @__omp_taskloop_taskred_add_i32.red.init
// CHECK:         store ptr @__omp_taskloop_taskred_add_i32.red.comb
// CHECK:         store i64 8
// CHECK:         store ptr @__omp_taskloop_taskred_mul_i64.red.init
// CHECK:         store ptr @__omp_taskloop_taskred_mul_i64.red.comb
// CHECK:         %[[DESC2:.+]] = call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 2, ptr %[[ARR2]])
// The descriptor is captured into structArg so the outlined task can reload it.
// CHECK:         store ptr %[[DESC2]], ptr %{{.+}}
// CHECK:         call void @__kmpc_taskloop(ptr {{.+}}, i32 {{.+}}, ptr {{.+}}, i32 1,
// CHECK:         call void @__kmpc_end_taskgroup(

// CHECK-LABEL: define internal void @taskloop_reduction_multi..omp_par(
// CHECK:         %[[BODY_GTID2:.+]] = call i32 @__kmpc_global_thread_num(
// Both get_th_data calls share the same body gtid; the descriptor argument
// must be a reloaded SSA value (not null).
// CHECK:         call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID2]], ptr %{{[^,]+}}, ptr %{{.+}})
// CHECK:         call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID2]], ptr %{{[^,]+}}, ptr %{{.+}})

// -----

// in_reduction on omp.taskloop.context nested inside an outer taskgroup
// task_reduction. No new __kmpc_taskgroup must be emitted for the taskloop
// itself (the user did not write reduction on it), and the get_th_data call
// must pass a NULL descriptor so the runtime walks up to the enclosing
// taskgroup.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskloop_inreduction(%x : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
  omp.taskgroup task_reduction(@add_i32 %x -> %tg : !llvm.ptr) {
    omp.taskloop.context in_reduction(@add_i32 %x -> %prv : !llvm.ptr) {
      omp.taskloop.wrapper {
        omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
          %v = llvm.load %prv : !llvm.ptr -> i32
          %s = llvm.add %v, %iv : i32
          llvm.store %s, %prv : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @taskloop_inreduction(
// Outer taskgroup opens once; we expect only ONE __kmpc_taskgroup for the
// outer construct (the taskloop itself must not open a second one).
// CHECK:         call void @__kmpc_taskgroup(
// CHECK-NOT:     call void @__kmpc_taskgroup(
// The outer descriptor is built; the taskloop must NOT build its own
// taskred_init.
// CHECK:         call ptr @__kmpc_taskred_init(
// CHECK-NOT:     call ptr @__kmpc_taskred_init(
// CHECK:         call void @__kmpc_taskloop(
// CHECK:         call void @__kmpc_end_taskgroup(

// In the outlined taskloop task body, the in_reduction lookup passes NULL
// as the descriptor argument so the runtime walks up enclosing taskgroups.
// CHECK-LABEL: define internal void @taskloop_inreduction..omp_par(
// CHECK:         call i32 @__kmpc_global_thread_num(
// CHECK:         call ptr @__kmpc_task_reduction_get_th_data(i32 %{{.+}}, ptr null, ptr %{{.+}})

// -----

// nogroup + in_reduction: the user wrote `nogroup` on the taskloop and only an
// in_reduction clause, so the translator must NOT open an implicit taskgroup
// and must NOT build a taskred descriptor for the taskloop itself; `nogroup`
// must be propagated to __kmpc_taskloop as 1, and the outlined body must look
// up the participant with a NULL descriptor so the runtime walks up.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskloop_nogroup_inreduction(%x : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
  omp.taskloop.context nogroup in_reduction(@add_i32 %x -> %prv : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        %v = llvm.load %prv : !llvm.ptr -> i32
        %s = llvm.add %v, %iv : i32
        llvm.store %s, %prv : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// Outer caller: no implicit taskgroup, no taskred_init, nogroup=1 to taskloop.
// CHECK-LABEL: define void @taskloop_nogroup_inreduction(
// CHECK-NOT:     call void @__kmpc_taskgroup(
// CHECK-NOT:     call ptr @__kmpc_taskred_init(
// CHECK-NOT:     call void @__kmpc_end_taskgroup(
// CHECK:         call void @__kmpc_taskloop(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 1,

// In the outlined task body, the in_reduction lookup uses a NULL descriptor.
// CHECK-LABEL: define internal void @taskloop_nogroup_inreduction..omp_par(
// CHECK:         call i32 @__kmpc_global_thread_num(
// CHECK:         call ptr @__kmpc_task_reduction_get_th_data(i32 %{{.+}}, ptr null, ptr %{{.+}})
