; RUN: opt -passes=lower-comment-string -S < %s | FileCheck %s

target triple = "powerpc64-ibm-aix"

@sccsid = internal global ptr @.str, align 8
@.str = private unnamed_addr constant [24 x i8] c"@(#) sccsid Version 1.0\00", align 1
@version = internal global [27 x i8] c"@(#) Copyright Version 2.0\00", align 1
@llvm.compiler.used = appending global [2 x ptr] [ptr @sccsid, ptr @version], section "llvm.metadata"

define void @foo() {
entry:
  ret void
}

define void @bar() {
entry:
  ret void
}

!loadtime_comment.vars = !{!1, !2}
!1 = !{!"sccsid"}
!2 = !{!"version"}

; CHECK: @sccsid = internal global ptr @.str, align {{[0-9]+}}
; CHECK: @.str = private unnamed_addr constant [24 x i8] c"@(#) sccsid Version 1.0\00", align {{[0-9]+}}
; CHECK: @version = internal global [27 x i8] c"@(#) Copyright Version 2.0\00", align {{[0-9]+}}
; CHECK: @llvm.compiler.used = appending global [2 x ptr] [ptr @sccsid, ptr @version], section "llvm.metadata"

; CHECK: define void @foo() !implicit.ref ![[REF1:[0-9]+]] !implicit.ref ![[REF2:[0-9]+]] {
; CHECK: define void @bar() !implicit.ref ![[REF1]] !implicit.ref ![[REF2]] {

; Verify that the generated implicit.ref metadata nodes point to the correct global variables.
; CHECK: ![[REF1]] = !{ptr @sccsid}
; CHECK: ![[REF2]] = !{ptr @version}
