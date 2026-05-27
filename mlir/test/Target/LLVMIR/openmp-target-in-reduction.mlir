// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// in_reduction on omp.target: the in_reduction variable is also captured
// into the target region as a map entry (the Flang front-end emits this
// implicit map). Inside the outlined target body the captured pointer is
// passed to __kmpc_task_reduction_get_th_data with a NULL descriptor;
// the runtime walks enclosing taskgroups to locate the matching
// task_reduction registration. The returned pointer is bound to the
// in_reduction region block argument so subsequent loads/stores inside
// the region use the private copy.

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

llvm.func @target_inreduction(%x : !llvm.ptr) {
  %m = omp.map.info var_ptr(%x : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
  omp.target in_reduction(@add_i32 %x -> %prv : !llvm.ptr) map_entries(%m -> %marg : !llvm.ptr) {
    %v = llvm.load %prv : !llvm.ptr -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %s = llvm.add %v, %c1 : i32
    llvm.store %s, %prv : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// The host stub forwards the captured pointer into the outlined target
// kernel.
// CHECK-LABEL: define void @target_inreduction(
// CHECK:         call void @__omp_offloading_{{.*}}_target_inreduction_{{.*}}(ptr %{{.+}}, ptr null)

// In the outlined target body the in_reduction private pointer is
// obtained from the runtime using the captured original pointer; that
// pointer is then the base of the load and store inside the region.
// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_target_inreduction_
// CHECK-SAME:    (ptr %[[CAPT:.+]], ptr %{{.+}})
// CHECK:         %[[GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[GTID]], ptr null, ptr %[[CAPT]])
// CHECK:         %[[LOADED:.+]] = load i32, ptr %[[PRIV]]
// CHECK:         %[[SUM:.+]] = add i32 %[[LOADED]], 1
// CHECK:         store i32 %[[SUM]], ptr %[[PRIV]]
