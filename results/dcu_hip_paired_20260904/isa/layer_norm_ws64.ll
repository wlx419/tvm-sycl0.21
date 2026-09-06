; ModuleID = 'TVMAMDGPUModule'
source_filename = "TVMAMDGPUModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa-hcc"

@shmem.0 = internal unnamed_addr addrspace(3) global float undef
@shmem.1.0 = internal unnamed_addr addrspace(3) global float undef

; Function Attrs: mustprogress nounwind willreturn
define dllexport amdgpu_kernel void @layer_norm_kernel(ptr addrspace(1) noalias nocapture writeonly %T_layer_norm, ptr addrspace(1) noalias nocapture readonly %ln1.bias, ptr addrspace(1) noalias nocapture readonly %ln1.weight, ptr addrspace(1) noalias nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %1 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %2 = shl nsw i32 %0, 6
  %3 = add nsw i32 %2, %1
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds float, ptr addrspace(1) %x, i64 %4
  %6 = load float, ptr addrspace(1) %5, align 4, !tbaa !4, !amdgpu.noclobber !7
  %v_x_red_temp.v0 = fadd float %6, 0.000000e+00
  %v_x_red_temp.v1 = tail call float @llvm.fmuladd.f32(float %6, float %6, float 0.000000e+00)
  %7 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %8 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %7)
  %9 = and i32 %8, 63
  %10 = icmp ugt i32 %9, 31
  %11 = shl i32 %8, 2
  %12 = add i32 %11, 128
  %13 = select i1 %10, i32 %11, i32 %12
  %14 = bitcast float %v_x_red_temp.v0 to i32
  %15 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %14)
  %16 = bitcast i32 %15 to float
  %17 = bitcast float %v_x_red_temp.v1 to i32
  %18 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %17)
  %19 = bitcast i32 %18 to float
  %20 = fadd float %v_x_red_temp.v0, %16
  %21 = fadd float %v_x_red_temp.v1, %19
  %22 = icmp ugt i32 %9, 47
  %23 = add i32 %11, 64
  %24 = select i1 %22, i32 %11, i32 %23
  %25 = bitcast float %20 to i32
  %26 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %24, i32 %25)
  %27 = bitcast i32 %26 to float
  %28 = bitcast float %21 to i32
  %29 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %24, i32 %28)
  %30 = bitcast i32 %29 to float
  %31 = fadd float %20, %27
  %32 = fadd float %21, %30
  %33 = icmp ugt i32 %9, 55
  %34 = add i32 %11, 32
  %35 = select i1 %33, i32 %11, i32 %34
  %36 = bitcast float %31 to i32
  %37 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %35, i32 %36)
  %38 = bitcast i32 %37 to float
  %39 = bitcast float %32 to i32
  %40 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %35, i32 %39)
  %41 = bitcast i32 %40 to float
  %42 = fadd float %31, %38
  %43 = fadd float %32, %41
  %44 = icmp ugt i32 %9, 59
  %45 = add i32 %11, 16
  %46 = select i1 %44, i32 %11, i32 %45
  %47 = bitcast float %42 to i32
  %48 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %46, i32 %47)
  %49 = bitcast i32 %48 to float
  %50 = bitcast float %43 to i32
  %51 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %46, i32 %50)
  %52 = bitcast i32 %51 to float
  %53 = fadd float %42, %49
  %54 = fadd float %43, %52
  %55 = icmp ugt i32 %9, 61
  %56 = add i32 %11, 8
  %57 = select i1 %55, i32 %11, i32 %56
  %58 = bitcast float %53 to i32
  %59 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %57, i32 %58)
  %60 = bitcast i32 %59 to float
  %61 = bitcast float %54 to i32
  %62 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %57, i32 %61)
  %63 = bitcast i32 %62 to float
  %64 = fadd float %53, %60
  %65 = fadd float %54, %63
  %66 = icmp ne i32 %9, 63
  %67 = zext i1 %66 to i32
  %68 = add nsw i32 %8, %67
  %69 = shl i32 %68, 2
  %70 = bitcast float %64 to i32
  %71 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %69, i32 %70)
  %72 = bitcast i32 %71 to float
  %73 = bitcast float %65 to i32
  %74 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %69, i32 %73)
  %75 = bitcast i32 %74 to float
  %76 = fadd float %64, %72
  %77 = fadd float %65, %75
  %78 = and i32 %11, -256
  %79 = bitcast float %76 to i32
  %80 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %78, i32 %79)
  %81 = bitcast float %77 to i32
  %82 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %78, i32 %81)
  %83 = icmp eq i32 %1, 0
  br i1 %83, label %if_then, label %if_end, !prof !8

if_then:                                          ; preds = %entry
  store i32 %80, ptr addrspace(3) @shmem.0, align 4, !tbaa !9
  store i32 %82, ptr addrspace(3) @shmem.1.0, align 4, !tbaa !20
  br label %if_end

if_end:                                           ; preds = %if_then, %entry
  tail call void @llvm.amdgcn.s.barrier()
  %84 = load float, ptr addrspace(3) @shmem.0, align 4, !tbaa !9
  %85 = fmul float %84, 1.562500e-02
  %86 = fmul float %85, %85
  %87 = load float, ptr addrspace(3) @shmem.1.0, align 4, !tbaa !20
  %88 = fmul float %87, 1.562500e-02
  %89 = fsub float %88, %86
  %90 = fadd float %89, 0x3EE4F8B580000000
  %91 = tail call float @llvm.sqrt.f32(float %90)
  %92 = fdiv float 1.000000e+00, %91
  %93 = fsub float %6, %85
  %94 = fmul float %93, %92
  %95 = sext i32 %1 to i64
  %96 = getelementptr inbounds float, ptr addrspace(1) %ln1.weight, i64 %95
  %97 = load float, ptr addrspace(1) %96, align 4, !tbaa !31, !amdgpu.noclobber !7
  %98 = getelementptr inbounds float, ptr addrspace(1) %ln1.bias, i64 %95
  %99 = load float, ptr addrspace(1) %98, align 4, !tbaa !33, !amdgpu.noclobber !7
  %100 = tail call float @llvm.fmuladd.f32(float %94, float %97, float %99)
  %101 = getelementptr inbounds float, ptr addrspace(1) %T_layer_norm, i64 %4
  store float %100, ptr addrspace(1) %101, align 4, !tbaa !35
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #2

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.sqrt.f32(float) #2

; Function Attrs: nofree nosync nounwind readnone
define weak dso_local i16 @__truncsfhf2(float %a0) local_unnamed_addr #6 section ".text.tvm.fp16.conv" {
b0:
  %v0 = bitcast float %a0 to i32
  %v1 = and i32 %v0, 2147483647
  %v2 = add nsw i32 %v1, -947912704
  %v3 = add nsw i32 %v1, -1199570944
  %v4 = icmp ult i32 %v2, %v3
  br i1 %v4, label %b1, label %b5

b1:                                               ; preds = %b0
  %v5 = lshr i32 %v0, 13
  %v6 = and i32 %v5, 65535
  %v8 = and i32 %v0, 8191
  %v9 = icmp ugt i32 %v8, 4096
  br i1 %v9, label %b2, label %b3

b2:                                               ; preds = %b1
  %v10 = add nuw nsw i32 %v6, -114687
  br label %b13

b3:                                               ; preds = %b1
  %v7 = add nuw nsw i32 %v6, -114688
  %v11 = icmp eq i32 %v8, 4096
  %v12 = and i32 %v7, 65535
  %v13 = and i32 %v5, 1
  %v14 = add nuw nsw i32 %v12, %v13
  %spec.select = select i1 %v11, i32 %v14, i32 %v7
  br label %b13

b5:                                               ; preds = %b0
  %v15 = icmp ugt i32 %v1, 2139095040
  br i1 %v15, label %b6, label %b7

b6:                                               ; preds = %b5
  %v16 = lshr i32 %v0, 13
  %v17 = and i32 %v16, 511
  %v18 = or i32 %v17, 32256
  br label %b13

b7:                                               ; preds = %b5
  %v19 = icmp ugt i32 %v1, 1199570943
  br i1 %v19, label %b13, label %b8

b8:                                               ; preds = %b7
  %v20 = icmp ult i32 %v1, 754974720
  br i1 %v20, label %b13, label %b9

b9:                                               ; preds = %b8
  %v21 = lshr i32 %v1, 23
  %v22 = sub nsw i32 113, %v21
  %v23 = and i32 %v0, 8388607
  %v24 = or i32 %v23, 8388608
  %v25 = add nsw i32 %v21, -81
  %v26 = shl i32 %v24, %v25
  %v27 = icmp ne i32 %v26, 0
  %v28 = lshr i32 %v24, %v22
  %v29 = zext i1 %v27 to i32
  %v30 = lshr i32 %v28, 13
  %v31 = and i32 %v28, 8191
  %v32 = or i32 %v31, %v29
  %v33 = icmp ugt i32 %v32, 4096
  br i1 %v33, label %b10, label %b11

b10:                                              ; preds = %b9
  %v34 = add nuw nsw i32 %v30, 1
  br label %b13

b11:                                              ; preds = %b9
  %v35 = icmp eq i32 %v32, 4096
  %v36 = and i32 %v30, 1
  %v37 = select i1 %v35, i32 %v36, i32 0
  %spec.select1 = add nuw nsw i32 %v37, %v30
  br label %b13

b13:                                              ; preds = %b11, %b3, %b10, %b8, %b7, %b6, %b2
  %v38 = phi i32 [ %v18, %b6 ], [ %v10, %b2 ], [ 31744, %b7 ], [ 0, %b8 ], [ %v34, %b10 ], [ %spec.select, %b3 ], [ %spec.select1, %b11 ]
  %v39 = lshr i32 %v0, 16
  %v40 = and i32 %v39, 32768
  %v41 = or i32 %v38, %v40
  %vlast = trunc i32 %v41 to i16
  ret i16 %vlast
}

; Function Attrs: nofree nosync nounwind readnone
define weak dso_local float @__extendhfsf2(i16 %a0) local_unnamed_addr #6 section ".text.tvm.fp16.conv" {
b0:
  %v1 = and i16 %a0, 32767
  %v2 = zext i16 %v1 to i32
  %v3 = add nsw i16 %v1, -1024
  %v4 = icmp ult i16 %v3, 30720
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  %v5 = shl nuw nsw i32 %v2, 13
  %v6 = add nuw nsw i32 %v5, 939524096
  br label %b6

b2:                                               ; preds = %b0
  %v7 = icmp ugt i16 %v1, 31743
  br i1 %v7, label %b3, label %b4

b3:                                               ; preds = %b2
  %v8 = shl nuw nsw i32 %v2, 13
  %v9 = or i32 %v8, 2139095040
  br label %b6

b4:                                               ; preds = %b2
  %v10 = icmp eq i16 %v1, 0
  br i1 %v10, label %b6, label %b5

b5:                                               ; preds = %b4
  %v11 = icmp ult i16 %v1, 256
  %v12 = lshr i32 %v2, 8
  %v13 = select i1 %v11, i32 %v2, i32 %v12
  %v14 = select i1 %v11, i32 32, i32 24
  %v15 = icmp ult i32 %v13, 16
  %v16 = lshr i32 %v13, 4
  %v17 = add nsw i32 %v14, -4
  %v18 = select i1 %v15, i32 %v13, i32 %v16
  %v19 = select i1 %v15, i32 %v14, i32 %v17
  %v20 = icmp ult i32 %v18, 4
  %v21 = lshr i32 %v18, 2
  %v22 = add nsw i32 %v19, -2
  %v23 = select i1 %v20, i32 %v18, i32 %v21
  %v24 = select i1 %v20, i32 %v19, i32 %v22
  %v25 = icmp ult i32 %v23, 2
  %v26 = sub nsw i32 0, %v23
  %v27 = select i1 %v25, i32 %v26, i32 -2
  %v28 = add nsw i32 %v27, %v24
  %v29 = add nsw i32 %v28, -8
  %v30 = shl i32 %v2, %v29
  %v31 = xor i32 %v30, 8388608
  %v32.neg = mul i32 %v28, -8388608
  %v33 = add i32 %v32.neg, 1124073472
  %v34 = or i32 %v31, %v33
  br label %b6

b6:                                               ; preds = %b5, %b4, %b3, %b1
  %v35 = phi i32 [ %v6, %b1 ], [ %v9, %b3 ], [ %v34, %b5 ], [ 0, %b4 ]
  %v36 = and i16 %a0, -32768
  %v37 = zext i16 %v36 to i32
  %v38 = shl nuw i32 %v37, 16
  %v39 = or i32 %v35, %v38
  %v40 = bitcast i32 %v39 to float
  ret float %v40
}

attributes #0 = { mustprogress nounwind willreturn "amdgpu-flat-work-group-size"="1,1024" "target-cpu"="gfx906" }
attributes #1 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #4 = { convergent mustprogress nofree nounwind readnone willreturn }
attributes #5 = { convergent mustprogress nounwind willreturn }
attributes #6 = { nofree nosync nounwind readnone "target-cpu"="gfx906" "target-features" }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 1}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 15.0.0 (http://10.15.3.7/dcutoolkit/driverruntime/llvm-project.git 340750feeda88c9c2ce8ad481b11d9aa7f033d39)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"0x55a288decac0", !6, i64 0}
!6 = !{!"tvm-tbaa"}
!7 = !{}
!8 = !{!"branch_weights", i32 1048576, i32 1}
!9 = !{!10, !10, i64 0}
!10 = !{!"0x55a288797180.w4.b0", !11, i64 0}
!11 = !{!"0x55a288797180.w8.b0", !12, i64 0}
!12 = !{!"0x55a288797180.w16.b0", !13, i64 0}
!13 = !{!"0x55a288797180.w32.b0", !14, i64 0}
!14 = !{!"0x55a288797180.w64.b0", !15, i64 0}
!15 = !{!"0x55a288797180.w128.b0", !16, i64 0}
!16 = !{!"0x55a288797180.w256.b0", !17, i64 0}
!17 = !{!"0x55a288797180.w512.b0", !18, i64 0}
!18 = !{!"0x55a288797180.w1024.b0", !19, i64 0}
!19 = !{!"0x55a288797180", !6, i64 0}
!20 = !{!21, !21, i64 0}
!21 = !{!"0x55a2888be5d0.w4.b0", !22, i64 0}
!22 = !{!"0x55a2888be5d0.w8.b0", !23, i64 0}
!23 = !{!"0x55a2888be5d0.w16.b0", !24, i64 0}
!24 = !{!"0x55a2888be5d0.w32.b0", !25, i64 0}
!25 = !{!"0x55a2888be5d0.w64.b0", !26, i64 0}
!26 = !{!"0x55a2888be5d0.w128.b0", !27, i64 0}
!27 = !{!"0x55a2888be5d0.w256.b0", !28, i64 0}
!28 = !{!"0x55a2888be5d0.w512.b0", !29, i64 0}
!29 = !{!"0x55a2888be5d0.w1024.b0", !30, i64 0}
!30 = !{!"0x55a2888be5d0", !6, i64 0}
!31 = !{!32, !32, i64 0}
!32 = !{!"0x55a288dedad0", !6, i64 0}
!33 = !{!34, !34, i64 0}
!34 = !{!"0x55a288dedb80", !6, i64 0}
!35 = !{!36, !36, i64 0}
!36 = !{!"0x55a288dedfc0", !6, i64 0}
