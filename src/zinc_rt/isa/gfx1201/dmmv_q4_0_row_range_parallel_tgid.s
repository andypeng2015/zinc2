.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_dmmv_q4_0_row_range_parallel_tgid
.type zinc_rt_dmmv_q4_0_row_range_parallel_tgid,@function

// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output f32 row-result pointer
//   s[4:5] = Q4_0 weight rows pointer
//   s6     = cols, multiple of 32
//   s7     = rows, multiple of 64
//   s8     = workgroup_id_x
//   v0     = workitem_id_x, row within the 64-row chunk
//
// One dispatch covers N adjacent 64-row chunks. Each workgroup owns one chunk,
// and each lane owns one row within that chunk.
zinc_rt_dmmv_q4_0_row_range_parallel_tgid:
    v_mov_b32_e32 v8, v0
    s_lshl_b32 s9, s8, 6
    v_add_nc_u32_e32 v8, s9, v8
    v_mov_b32_e32 v1, 0
    s_lshr_b32 s10, s6, 5

    v_mul_u32_u24_e32 v9, s10, v8
    v_mul_u32_u24_e32 v9, 18, v9
    s_mov_b32 s11, 0

block_loop:
    s_cmp_ge_u32 s11, s10
    s_cbranch_scc1 store_row

    s_mul_i32 s13, s11, 18
    v_add_nc_u32_e32 v10, s13, v9
    global_load_ushort v2, v10, s[4:5]

    s_mul_i32 s14, s11, 32
    s_mov_b32 s15, 0
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16_e32 v2, v2

j_loop:
    s_cmp_ge_u32 s15, 16
    s_cbranch_scc1 next_block

    s_add_u32 s16, s13, 2
    s_add_u32 s16, s16, s15
    v_add_nc_u32_e32 v11, s16, v9
    global_load_ubyte v3, v11, s[4:5]

    s_add_u32 s17, s14, s15
    s_lshl_b32 s17, s17, 2
    v_mov_b32_e32 v12, s17
    global_load_b32 v6, v12, s[0:1]

    s_add_u32 s18, s14, s15
    s_add_u32 s18, s18, 16
    s_lshl_b32 s18, s18, 2
    v_mov_b32_e32 v13, s18
    global_load_b32 v7, v13, s[0:1]

    s_waitcnt vmcnt(0)
    v_and_b32_e32 v4, 0x0f, v3
    v_lshrrev_b32_e32 v5, 4, v3
    v_add_nc_u32_e32 v4, -8, v4
    v_add_nc_u32_e32 v5, -8, v5
    v_cvt_f32_i32_e32 v4, v4
    v_cvt_f32_i32_e32 v5, v5
    v_mul_f32_e32 v4, v2, v4
    v_fmac_f32_e32 v1, v4, v6
    v_mul_f32_e32 v5, v2, v5
    v_fmac_f32_e32 v1, v5, v7

    s_add_u32 s15, s15, 1
    s_branch j_loop

next_block:
    s_add_u32 s11, s11, 1
    s_branch block_loop

store_row:
    v_lshlrev_b32_e32 v14, 2, v8
    global_store_b32 v14, v1, s[2:3]

    s_branch done

done:
    s_nop 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
