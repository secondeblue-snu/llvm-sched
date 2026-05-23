// RUN: %clang_cc1 -O2 -triple powerpc-ibm-aix -mloadtime-comment-vars=sccsid,version,build_number -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-ibm-aix -mloadtime-comment-vars=sccsid,version,build_number -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// String pointer 
static char *sccsid = "@(#) sccsid Version 1.0";

// String array 
static char version[] = "@(#) Copyright Version 2.0";

// Const string (Not in CLI list, should NOT be emitted)
static const char *copyright = "@(#) Copyright 2026";

// Integer (In CLI list but invalid type, should NOT be emitted)
static int build_number = 12345;

// Struct (not in CLI list and invalid type, NOT emitted)
struct build_info {
    int major;
    int minor;
} static build_data = {1, 0};

void foo() {}

// CHECK: @sccsid = internal global ptr @.str, align {{[0-9]+}}
// CHECK: @.str = private unnamed_addr constant [24 x i8] c"@(#) sccsid Version 1.0\00", align {{[0-9]+}}
// CHECK: @version = internal global [27 x i8] c"@(#) Copyright Version 2.0\00", align {{[0-9]+}}
// CHECK: @llvm.compiler.used = appending global [2 x ptr] [ptr @sccsid, ptr @version], section "llvm.metadata"

// Ensure unrequested/invalid variables are not emitted
// CHECK-NOT: @copyright
// CHECK-NOT: @build_number
// CHECK-NOT: @build_data

// Verify named metadata contains the preserved variable names
// CHECK: !loadtime_comment.vars = !{![[MD_SCC:[0-9]+]], ![[MD_VER:[0-9]+]]}
// CHECK: ![[MD_SCC]] = !{!"sccsid"}
// CHECK: ![[MD_VER]] = !{!"version"}