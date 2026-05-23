//===-- LowerCommentStringPass.cpp - Lower Comment string metadata -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass processes copyright and variable metadata for AIX, handling two
// distinct mechanisms:
//
// 1. #pragma comment(copyright, "...") - TU-wide copyright strings
// 2. -mloadtime-comment-vars=<names> - User-specified global variables
//
// Both types of information must be preserved in the final object file and
// survive optimization passes including DCE and LTO.
//
// === #pragma comment(copyright, "...") ===
//
// Clang emits module-level metadata for copyright pragmas:
//
//     !comment_string.loadtime = !{!"Copyright ..."}
//
// This pass materializes the metadata into a concrete global variable:
//
//   1. Creates a null-terminated, internal constant string global
//      `__loadtime_comment_str` containing the copyright text with section
//      attribute "__loadtime_comment". The backend emits this to a special
//      section in the object file.
//
//   2. Marks the global in `llvm.compiler.used` to prevent removal by
//      optimization passes.
//
//   3. Attaches `!implicit.ref` metadata to every defined function,
//      referencing the global. The PowerPC AIX backend emits a `.ref`
//      directive for each reference, creating relocations that prevent the
//      linker from discarding the string.
//
// === -mloadtime-comment-vars=<names> ===
//
// Clang stores the names of user-specified global variables (e.g., char
// *sccsid, char version[]) in module-level metadata:
//
//     !loadtime_comment.vars = !{!{!"sccsid"}, !{!"version"}}
//
// This pass:
//
//   1. Reads the variable names from the metadata and looks up each global
//      by name using M.getNamedGlobal().
//
//   2. Attaches `!implicit.ref` metadata to every defined function,
//      referencing each tagged global. This ensures the variables survive
//      optimization and linking.
//
// === Output Example ===
//
// Input IR:
//
//   @sccsid = internal global ptr @.str, align 8
//   @.str = private unnamed_addr constant [24 x i8] c"@(#) sccsid
//   Version 1.0\00", align 1
//   @llvm.compiler.used = appending global [1 x ptr] [ptr @sccsid], section
//   "llvm.metadata"
//   !comment_string.loadtime = !{!1}
//   !loadtime_comment.vars = !{!2}
//   !1 = !{!"Pragma comment copyright"}
//   !2 = !{!"sccsid"}
//
// Output IR:
//   @sccsid = internal global ptr @.str, align 8
//   @.str = private unnamed_addr constant [24 x i8] c"@(#) sccsid
//   Version 1.0\00", align 1
//   @__loadtime_comment_str = internal unnamed_addr constant [25 x i8]
//   c"Pragma comment copyright\00", section "__loadtime_comment", align 1
//   @llvm.compiler.used = appending global [2 x ptr] [ptr @sccsid, ptr
//   @__loadtime_comment_str], section "llvm.metadata"
//
//   define void @foo() !implicit.ref !1 !implicit.ref !2 {
//   entry:
//     ret void
//   }
//
//   !1 = !{ptr @__loadtime_comment_str}
//   !2 = !{ptr @sccsid}
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerCommentStringPass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "lower-comment-string"

using namespace llvm;

static cl::opt<bool>
    DisableCopyrightMetadata("disable-lower-comment-string", cl::ReallyHidden,
                             cl::desc("Disable LowerCommentString pass."),
                             cl::init(false));

static bool isSupportedTarget(const Module &M) {
  Triple T{M.getTargetTriple()};
  return T.isOSAIX();
}

PreservedAnalyses LowerCommentStringPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  if (DisableCopyrightMetadata || !isSupportedTarget(M))
    return PreservedAnalyses::all();

  LLVMContext &Ctx = M.getContext();

  // This pass processes two types of copyright/identifying information:
  // 1. A single TU-wide copyright string from #pragma comment(copyright, "...")
  // 2. Multiple user-specified variables from -mloadtime-comment-vars=...
  //
  // Both need implicit references from every function to survive DCE and LTO.
  // Collect all copyright globals, then create implicit references
  // from every function definition to each global. This forces the backend
  // to treat them as reachable and preserve them in the final object file.
  SmallVector<GlobalValue *, 4> CopyrightGlobals;

  // =========================================================================
  // Process #pragma comment(copyright, "...") - at most one per TU
  // =========================================================================
  // Frontend emits module-level metadata:
  //   !comment_string.loadtime = !{!0}
  //   !0 = !{!"Copyright text here"}
  //
  // We materialize this as a global string in the __loadtime_comment section,
  // which linkers recognize and include in the object file's loadtime
  // comment area.
  NamedMDNode *MD = M.getNamedMetadata("comment_string.loadtime");
  if (MD && MD->getNumOperands() > 0) {
    MDNode *MdNode = MD->getOperand(0);
    if (MdNode && MdNode->getNumOperands() > 0) {
      auto *MdString = dyn_cast_or_null<MDString>(MdNode->getOperand(0));
      if (MdString && !MdString->getString().empty()) {
        StringRef Text = MdString->getString();
        // Create a null-terminated string constant in the special section.
        Constant *StrInit =
            ConstantDataArray::getString(Ctx, Text, /*AddNull*/ true);
        // The global variable should be internal, constant, and TU-local.
        // This avoids duplicate symbol issues across TUs.
        auto *StrGV = new GlobalVariable(M, StrInit->getType(),
                                         /*isConstant=*/true,
                                         GlobalValue::InternalLinkage, StrInit,
                                         /*Name=*/"__loadtime_comment_str");
        StrGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
        StrGV->setAlignment(Align(1));
        // Backend recognizes this section and emits it to .loadtime_comment.
        StrGV->setSection("__loadtime_comment");
        // Prevent removal by optimizer passes (but not sufficient for linker).
        appendToCompilerUsed(M, {StrGV});
        // Add to list - will get implicit refs from all functions below.
        CopyrightGlobals.push_back(StrGV);
      }
    }
    // Clean up the metadata as we have consumed it.
    MD->eraseFromParent();
  }

  // =========================================================================
  // Process -mloadtime-comment-vars=sccsid,version,... (CLI flag)
  // =========================================================================
  // Frontend stores variable names in named metadata:
  //   !loadtime_comment.vars = !{!{!"sccsid"}, !{!"version"}}
  //
  // We look each name up by M.getNamedGlobal() rather than walking globals
  // looking for per-global metadata, because per-global metadata is droppable
  // and may be stripped by optimization passes before this pass runs.
  NamedMDNode *VarsMD = M.getNamedMetadata("loadtime_comment.vars");
  if (VarsMD) {
    for (unsigned I = 0, E = VarsMD->getNumOperands(); I < E; ++I) {
      MDNode *Entry = VarsMD->getOperand(I);
      if (!Entry || Entry->getNumOperands() == 0)
        continue;

      auto *VarName = dyn_cast_or_null<MDString>(Entry->getOperand(0));
      if (!VarName || VarName->getString().empty())
        continue;

      GlobalValue *GV = M.getNamedGlobal(VarName->getString());
      if (!GV || GV->isDeclaration())
        continue;

      appendToCompilerUsed(M, {GV});

      CopyrightGlobals.push_back(GV);
    }
    VarsMD->eraseFromParent();
  }

  // =========================================================================
  // Create implicit references from every function to each global
  // =========================================================================
  // Each implicit.ref node references exactly ONE global. Multiple nodes
  // can be attached to a single function (e.g., !implicit.ref !1, !implicit.ref
  // !2).
  auto AddImplicitRef = [&](Function &F, GlobalValue *GV) {
    if (F.isDeclaration())
      return;
    // Create metadata: !N = !{ptr @global_variable}
    Metadata *Ops[] = {ConstantAsMetadata::get(GV)};
    MDNode *NewMD = MDNode::get(Ctx, Ops);
    // Attach to function - addMetadata allows multiple !implicit.ref nodes per
    // function, one for each copyright global.
    F.addMetadata(LLVMContext::MD_implicit_ref, *NewMD);

    LLVM_DEBUG(dbgs() << "[copyright] attached implicit.ref to function: "
                      << F.getName() << " for global: " << GV->getName()
                      << "\n");
  };

  // Apply implicit references: for each global, mark all functions as users.
  if (!CopyrightGlobals.empty()) {
    for (GlobalValue *GV : CopyrightGlobals) {
      for (Function &F : M)
        AddImplicitRef(F, GV);
    }
  }

  LLVM_DEBUG(dbgs() << "[copyright] processed " << CopyrightGlobals.size()
                    << " copyright globals\n");

  return PreservedAnalyses::all();
}
