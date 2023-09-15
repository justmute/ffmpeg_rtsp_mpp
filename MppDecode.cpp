//
// Created by LX on 2020/4/25.
//

#include "MppDecode.h"
#include <stdio.h>
#include <unistd.h>
#include <opencv2/imgproc/types_c.h>

#define MPP_FREE(ptr) do { if (ptr) mpp_free(ptr); ptr = NULL; } while(0)
#define MPP_ABS(x)              ((x) < (0) ? -(x) : (x))

#define MPP_MAX(a, b)           ((a) > (b) ? (a) : (b))
#define MPP_MAX3(a, b, c)       MPP_MAX(MPP_MAX(a,b),c)
#define MPP_MAX4(a, b, c, d)    MPP_MAX((a), MPP_MAX3((b), (c), (d)))

#define MPP_MIN(a,b)            ((a) > (b) ? (b) : (a))
#define MPP_MIN3(a,b,c)         MPP_MIN(MPP_MIN(a,b),c)
#define MPP_MIN4(a, b, c, d)    MPP_MIN((a), MPP_MIN3((b), (c), (d)))

#define MPP_DIV(a, b)           ((b) ? (a) / (b) : (a))

#define MPP_CLIP3(l, h, v)      ((v) < (l) ? (l) : ((v) > (h) ? (h) : (v)))
#define MPP_SIGN(a)             ((a) < (0) ? (-1) : (1))
#define MPP_DIV_SIGN(a, b)      (((a) + (MPP_SIGN(a) * (b)) / 2) / (b))

#define MPP_SWAP(type, a, b)    do {type SWAP_tmp = b; b = a; a = SWAP_tmp;} while(0)
#define MPP_ARRAY_ELEMS(a)      (sizeof(a) / sizeof((a)[0]))
#define MPP_ALIGN(x, a)         (((x)+(a)-1)&~((a)-1))
#define MPP_VSWAP(a, b)         { a ^= b; b ^= a; a ^= b; }

#define MPP_RB16(x)  ((((const RK_U8*)(x))[0] << 8) | ((const RK_U8*)(x))[1])
#define MPP_WB16(p, d) do { \
        ((RK_U8*)(p))[1] = (d); \
        ((RK_U8*)(p))[0] = (d)>>8; } while(0)

#define MPP_RL16(x)  ((((const RK_U8*)(x))[1] << 8) | \
                     ((const RK_U8*)(x))[0])
#define MPP_WL16(p, d) do { \
        ((RK_U8*)(p))[0] = (d); \
        ((RK_U8*)(p))[1] = (d)>>8; } while(0)

#define MPP_RB32(x)  ((((const RK_U8*)(x))[0] << 24) | \
                     (((const RK_U8*)(x))[1] << 16) | \
                     (((const RK_U8*)(x))[2] <<  8) | \
                     ((const RK_U8*)(x))[3])
#define MPP_WB32(p, d) do { \
        ((RK_U8*)(p))[3] = (d); \
        ((RK_U8*)(p))[2] = (d)>>8; \
        ((RK_U8*)(p))[1] = (d)>>16; \
        ((RK_U8*)(p))[0] = (d)>>24; } while(0)

#define MPP_RL32(x) ((((const RK_U8*)(x))[3] << 24) | \
                    (((const RK_U8*)(x))[2] << 16) | \
                    (((const RK_U8*)(x))[1] <<  8) | \
                    ((const RK_U8*)(x))[0])
#define MPP_WL32(p, d) do { \
        ((RK_U8*)(p))[0] = (d); \
        ((RK_U8*)(p))[1] = (d)>>8; \
        ((RK_U8*)(p))[2] = (d)>>16; \
        ((RK_U8*)(p))[3] = (d)>>24; } while(0)

#define MPP_RB64(x)  (((RK_U64)((const RK_U8*)(x))[0] << 56) | \
                     ((RK_U64)((const RK_U8*)(x))[1] << 48) | \
                     ((RK_U64)((const RK_U8*)(x))[2] << 40) | \
                     ((RK_U64)((const RK_U8*)(x))[3] << 32) | \
                     ((RK_U64)((const RK_U8*)(x))[4] << 24) | \
                     ((RK_U64)((const RK_U8*)(x))[5] << 16) | \
                     ((RK_U64)((const RK_U8*)(x))[6] <<  8) | \
                     (RK_U64)((const RK_U8*)(x))[7])
#define MPP_WB64(p, d) do { \
        ((RK_U8*)(p))[7] = (d);     \
        ((RK_U8*)(p))[6] = (d)>>8;  \
        ((RK_U8*)(p))[5] = (d)>>16; \
        ((RK_U8*)(p))[4] = (d)>>24; \
        ((RK_U8*)(p))[3] = (d)>>32; \
        ((RK_U8*)(p))[2] = (d)>>40; \
        ((RK_U8*)(p))[1] = (d)>>48; \
        ((RK_U8*)(p))[0] = (d)>>56; } while(0)

#define MPP_RL64(x)  (((RK_U64)((const RK_U8*)(x))[7] << 56) | \
                     ((RK_U64)((const RK_U8*)(x))[6] << 48) | \
                     ((RK_U64)((const RK_U8*)(x))[5] << 40) | \
                     ((RK_U64)((const RK_U8*)(x))[4] << 32) | \
                     ((RK_U64)((const RK_U8*)(x))[3] << 24) | \
                     ((RK_U64)((const RK_U8*)(x))[2] << 16) | \
                     ((RK_U64)((const RK_U8*)(x))[1] <<  8) | \
                     (RK_U64)((const RK_U8*)(x))[0])
#define MPP_WL64(p, d) do { \
        ((RK_U8*)(p))[0] = (d);     \
        ((RK_U8*)(p))[1] = (d)>>8;  \
        ((RK_U8*)(p))[2] = (d)>>16; \
        ((RK_U8*)(p))[3] = (d)>>24; \
        ((RK_U8*)(p))[4] = (d)>>32; \
        ((RK_U8*)(p))[5] = (d)>>40; \
        ((RK_U8*)(p))[6] = (d)>>48; \
        ((RK_U8*)(p))[7] = (d)>>56; } while(0)

#define MPP_RB24(x)  ((((const RK_U8*)(x))[0] << 16) | \
                     (((const RK_U8*)(x))[1] <<  8) | \
                     ((const RK_U8*)(x))[2])
#define MPP_WB24(p, d) do { \
        ((RK_U8*)(p))[2] = (d); \
        ((RK_U8*)(p))[1] = (d)>>8; \
        ((RK_U8*)(p))[0] = (d)>>16; } while(0)

#define MPP_RL24(x)  ((((const RK_U8*)(x))[2] << 16) | \
                     (((const RK_U8*)(x))[1] <<  8) | \
                     ((const RK_U8*)(x))[0])

#define MPP_WL24(p, d) do { \
        ((RK_U8*)(p))[0] = (d); \
        ((RK_U8*)(p))[1] = (d)>>8; \
        ((RK_U8*)(p))[2] = (d)>>16; } while(0)


void dump_mpp_frame_to_file(MppFrame frame, FILE *fp)
{
	RK_U32 width    = 0;
    RK_U32 height   = 0;
    RK_U32 h_stride = 0;
    RK_U32 v_stride = 0;
	MppFrameFormat fmt  = MPP_FMT_YUV420SP;
    MppBuffer buffer    = NULL;
    RK_U8 *base = NULL;

	if (NULL == fp || NULL == frame)
		return;

    width    = mpp_frame_get_width(frame);
    height   = mpp_frame_get_height(frame);
    h_stride = mpp_frame_get_hor_stride(frame);
    v_stride = mpp_frame_get_ver_stride(frame);
	fmt		 = mpp_frame_get_fmt(frame);
    buffer   = mpp_frame_get_buffer(frame);
	printf("MppFrame width=%d height=%d hor_stride=%d v_stride=%d fmt=%d\n",
			width,height,h_stride,v_stride,fmt);

    if (NULL == buffer)
		return;

	base = (RK_U8 *)mpp_buffer_get_ptr(buffer);
    
	// ------------ log ------------
	RK_U32 frame_buf_size = mpp_frame_get_buf_size(frame);
    size_t frame_buffer_size = mpp_buffer_get_size(buffer);
    printf("mpp_frame_get_buf_size(frame) = %d mpp_buffer_get_size(buffer) = %d\n",
			frame_buf_size,frame_buffer_size);
	// ------------ log ------------

	//if (MPP_FRAME_FMT_IS_RGB(fmt) && MPP_FRAME_FMT_IS_LE(fmt))
	//{
	//	fmt &= MPP_FRAME_FMT_MASK;
	//}

	switch (fmt & MPP_FRAME_FMT_MASK)
	{
	case MPP_FMT_YUV422SP:
	{
		// YUV422SP -> YUV422P for better display
		RK_U32 i,j;
		RK_U8 *base_y = base;
		RK_U8 *base_c = base + h_stride * v_stride;
		//RK_U8 *tmp = mpp_malloc(RK_U8, h_stride * height * 2);
		RK_U8 *tmp = (RK_U8*)av_mallocz(sizeof(RK_U8) * h_stride * height * 2);
		RK_U8 *tmp_u = tmp;
		RK_U8 *tmp_v = tmp + width * height / 2;

		for (i = 0; i < height; i++, base_y += h_stride)
			fwrite(base_y, 1, width, fp);

		for (i = 0; i < height; i++, base_c += h_stride)
		{
			for (j = 0; j < width/2; j++)
			{
				tmp_u[j] = base_c[2*j+0];
				tmp_v[j] = base_c[2*j+1];
			}
			tmp_u += width/2;
			tmp_v += width/2;
		}
		
		fwrite(tmp, 1, width*height, fp);
		//mpp_free(tmp);
		av_free(tmp);
	}	
		break;
	case MPP_FMT_YUV420SP_VU:
	case MPP_FMT_YUV420SP:
	{
		RK_U32 i;
		RK_U8 *base_y = base;
		RK_U8 *base_c = base + h_stride * v_stride;
		
		for (i = 0; i < height; i++, base_y += h_stride)
		{
			fwrite(base_y, 1, width, fp);
		}
		for (i = 0; i < height/2; i++, base_c += h_stride)
		{
			fwrite(base_c, 1, width, fp);
		}
	}
		break;
	case MPP_FMT_YUV420P:
	{
		RK_U32 i;
		RK_U8 *base_y = base;
		RK_U8 *base_c = base + h_stride * v_stride;

		for(i = 0; i < height; i++, base_y += h_stride)
		{
			fwrite(base_y, 1, width, fp);
		}

		for(i = 0; i < height / 2; i++, base_c += h_stride/2)
		{
			fwrite(base_c, 1, width/2, fp);
		}

		for(i = 0; i < height / 2; i++, base_c += h_stride/2)
		{
			fwrite(base_c, 1, width/2, fp);
		}
	}
		break;
	case MPP_FMT_YUV420SP_10BIT:
	{
		RK_U32 i,k;
		RK_U8 *base_y = base;
		RK_U8 *base_c = base + h_stride * v_stride;
		//RK_U8 *tmp_line = (RK_U8*)mpp_malloc(RK_U16, width);
		RK_U8 *tmp_line = (RK_U8*)av_mallocz(sizeof(RK_U16) * width);

		if (!tmp_line)
		{
			printf("tmp_line malloc fail!");
			return;
		}

		for (i = 0; i < height; i++, base_y += h_stride)
		{
			for (k = 0; k < width/8; k++)
			{
				RK_U16 *pix = (RK_U16*)(tmp_line + k*16);
				RK_U16 *base_u16 = (RK_U16*)(base_y + k*10);

				pix[0] = base_u16[0] & 0x03FF;
				pix[1] = (base_u16[0] & 0xFC00) >> 10 | (base_u16[1] & 0x000F) << 6;
				pix[2] = (base_u16[1] & 0x3FF0) >> 4;
				pix[3] = (base_u16[1] & 0xC000) >> 14 | (base_u16[2] & 0x00FF) << 2;
				pix[4] = (base_u16[2] & 0xFF00) >> 8  | (base_u16[3] & 0x0003) << 8;
				pix[5] = (base_u16[3] & 0x0FFC) >> 2;
				pix[6] = (base_u16[3] & 0xF000) >> 12 | (base_u16[4] & 0x003F) << 4;
				pix[7] = (base_u16[4] & 0xFFC0) >> 6;
			}
			fwrite(tmp_line, width * sizeof(RK_U16), 1, fp);
		}

		for (i = 0; i < height/2; i++, base_c += h_stride)
		{
			for (k = 0; k < width/8; k++)
			{
				RK_U16 *pix = (RK_U16*)(tmp_line + k*16);
				RK_U16 *base_u16 = (RK_U16*)(base_c + k*10);

				pix[0] = base_u16[0] & 0x03FF;
				pix[1] = (base_u16[0] & 0xFC00) >> 10 | (base_u16[1] & 0x000F) << 6;
				pix[2] = (base_u16[1] & 0x3FF0) >> 4;
				pix[3] = (base_u16[1] & 0xC000) >> 14 | (base_u16[2] & 0x00FF) << 2;
				pix[4] = (base_u16[2] & 0xFF00) >> 8  | (base_u16[3] & 0x0003) << 8;
				pix[5] = (base_u16[3] & 0x0FFC) >> 2;
				pix[6] = (base_u16[3] & 0xF000) >> 12 | (base_u16[4] & 0x003F) << 4;
				pix[7] = (base_u16[4] & 0xFFC0) >> 6;
			}
			fwrite(tmp_line, width * sizeof(RK_U16), 1, fp);
		}

		//MPP_free(tmp_line);
		av_free(tmp_line);
	}	
		break;
	case MPP_FMT_YUV444SP:
	{
		// YUV444SP -> YUV444P for better display
		RK_U32 i,j;
		RK_U8 *base_y = base;
		RK_U8 *base_c = base + h_stride * v_stride;
		//RK_U8 *tmp = mpp_malloc(RK_U8, h_stride * height * 2);
		RK_U8 *tmp = (RK_U8*)av_mallocz(sizeof(RK_U8) * h_stride * height * 2);
		RK_U8 *tmp_u = tmp;
		RK_U8 *tmp_v = tmp + width * height;

		for (i = 0; i < height; i++, base_y += h_stride)
			fwrite(base_y, 1, width, fp);

		for (i = 0; i < height; i++, base_c += h_stride * 2)
		{
			for (j = 0; j < width; j++)
			{
				tmp_u[j] = base_c[2 * j + 0];
				tmp_v[j] = base_c[2 * j + 1];
			}
			tmp_u += width;
			tmp_v += width;
		}
		
		fwrite(tmp, 1, width*height*2, fp);
		//mpp_free(tmp);
		av_free(tmp);
	}
		break;
	case MPP_FMT_YUV400:
	{
		RK_U32 i;
		RK_U8 *base_y = base;
		//RK_U8 *tmp = mpp_malloc(RK_U8, h_stride * height);
		RK_U8 *tmp = (RK_U8*)av_mallocz(sizeof(RK_U8) * h_stride * height);

		for (i = 0; i < height; i++, base_y += h_stride)
			fwrite(base_y, 1, width, fp);
		
		//mpp_free(tmp);
		av_free(tmp);
	}
		break;
	case MPP_FMT_ARGB8888:
	case MPP_FMT_ABGR8888:
	case MPP_FMT_BGRA8888:
	case MPP_FMT_RGBA8888:
	{
		RK_U32 i;
		RK_U8 *base_y = base;
		//RK_U8 *tmp = mpp_malloc(RK_U8, width*height*4);
		RK_U8 *tmp = (RK_U8*)av_mallocz(sizeof(RK_U8) * width*height*4);

		for (i=0; i < height; i++, base_y += h_stride)
			fwrite(base_y, 1, width*4, fp);

		//mpp_free(tmp);
		av_free(tmp);
	}
		break;
	case MPP_FMT_RGB565:
	case MPP_FMT_BGR565:
	case MPP_FMT_RGB555:
	case MPP_FMT_BGR555:
	case MPP_FMT_RGB444:
	case MPP_FMT_BGR444:
	{
		RK_U32 i;
		RK_U8 *base_y = base;
		//RK_U8 *tmp = mpp_malloc(RK_U8, width * height * 2);
		RK_U8 *tmp = (RK_U8*)av_mallocz(sizeof(RK_U8) * width * height * 2);

		for (i = 0; i < height; i++,base_y += h_stride)
			fwrite(base_y, 1, width * 2, fp);

		//mpp_free(tmp);
		av_free(tmp);
	}
		break;
	default:
		printf("not supported format %d\n",fmt);
		break;
	}
	
    //RK_U32 i;
    //RK_U8 *base_y = base;
    //RK_U8 *base_c = base + h_stride * v_stride;
//#ifdef YUV420sp
    //保存为YUV420sp格式
    //for (i = 0; i < height; i++, base_y += h_stride)
    //{
    //    fwrite(base_y, 1, width, fp);
    //}
    //for (i = 0; i < height / 2; i++, base_c += h_stride)
    //{
    //    fwrite(base_c, 1, width, fp);
    //}
//#else
    //保存为YUV420p格式
    //for(i = 0; i < height; i++, base_y += h_stride)
    //{
    //    fwrite(base_y, 1, width, fp);
    //}
    //for(i = 0; i < height * width / 2; i+=2)
    //{
    //    fwrite((base_c + i), 1, 1, fp);
    //}
    //for(i = 1; i < height * width / 2; i+=2)
    //{
    //    fwrite((base_c + i), 1, 1, fp);
    //}
//#endif
}

int decode_simple(MpiDecLoopData *data, AVPacket *av_packet )
{
    RK_U32 pkt_done = 0;
    RK_U32 pkt_eos  = 0;
    MPP_RET ret = MPP_OK;

    MppCtx ctx  = data->ctx;
    MppApi *mpi = data->mpi;
    
	MppPacket packet = NULL;
    
	size_t read_size = 0;
    size_t packet_size = data->packet_size;

    ret = mpp_packet_init(&packet, av_packet->data, av_packet->size);
    mpp_packet_set_pts(packet, av_packet->pts);


    do {
        RK_U32 frm_eos = 0;
        RK_S32 times = 5;
        
		// send the packet first if packet is not done
        if (!pkt_done) {
            ret = mpi->decode_put_packet(ctx, packet);
            if (MPP_OK == ret)
                pkt_done = 1;
        }

        // then get all available frame and release
        do {
            RK_S32 get_frm = 0;
			MppFrame frame = NULL;

        try_again:
            ret = mpi->decode_get_frame(ctx, &frame);
            if (MPP_ERR_TIMEOUT == ret) {
                if (times > 0) {
                    times--;
                    usleep(2*1000);
                    goto try_again;
                }
                printf("decode_get_frame failed too much time\n");
            }
            if (ret) {
                printf("decode_get_frame failed ret %d\n", ret);
                break;
            }

            if (frame) {
                if (mpp_frame_get_info_change(frame)) {
                    RK_U32 width = mpp_frame_get_width(frame);
                    RK_U32 height = mpp_frame_get_height(frame);
                    RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
                    RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);
                    RK_U32 buf_size = mpp_frame_get_buf_size(frame);

                    printf("decode_get_frame get info changed found\n");
                    printf("decoder require buffer w:h [%d:%d] stride [%d:%d] buf_size %d\n",
                            width, height, hor_stride, ver_stride, buf_size);

					/*
                     * NOTE: We can choose decoder's buffer mode here.
                     * There are three mode that decoder can support:
                     *
                     * Mode 1: Pure internal mode
                     * In the mode user will NOT call MPP_DEC_SET_EXT_BUF_GROUP
                     * control to decoder. Only call MPP_DEC_SET_INFO_CHANGE_READY
                     * to let decoder go on. Then decoder will use create buffer
                     * internally and user need to release each frame they get.
                     *
                     * Advantage:
                     * Easy to use and get a demo quickly
                     * Disadvantage:
                     * 1. The buffer from decoder may not be return before
                     * decoder is close. So memroy leak or crash may happen.
                     * 2. The decoder memory usage can not be control. Decoder
                     * is on a free-to-run status and consume all memory it can
                     * get.
                     * 3. Difficult to implement zero-copy display path.
                     *
                     * Mode 2: Half internal mode
                     * This is the mode current test code using. User need to
                     * create MppBufferGroup according to the returned info
                     * change MppFrame. User can use mpp_buffer_group_limit_config
                     * function to limit decoder memory usage.
                     *
                     * Advantage:
                     * 1. Easy to use
                     * 2. User can release MppBufferGroup after decoder is closed.
                     *    So memory can stay longer safely.
                     * 3. Can limit the memory usage by mpp_buffer_group_limit_config
                     * Disadvantage:
                     * 1. The buffer limitation is still not accurate. Memory usage
                     * is 100% fixed.
                     * 2. Also difficult to implement zero-copy display path.
                     *
                     * Mode 3: Pure external mode
                     * In this mode use need to create empty MppBufferGroup and
                     * import memory from external allocator by file handle.
                     * On Android surfaceflinger will create buffer. Then
                     * mediaserver get the file handle from surfaceflinger and
                     * commit to decoder's MppBufferGroup.
                     *
                     * Advantage:
                     * 1. Most efficient way for zero-copy display
                     * Disadvantage:
                     * 1. Difficult to learn and use.
                     * 2. Player work flow may limit this usage.
                     * 3. May need a external parser to get the correct buffer
                     * size for the external allocator.
                     *
                     * The required buffer size caculation:
                     * hor_stride * ver_stride * 3 / 2 for pixel data
                     * hor_stride * ver_stride / 2 for extra info
                     * Total hor_stride * ver_stride * 2 will be enough.
                     *
                     * For H.264/H.265 20+ buffers will be enough.
                     * For other codec 10 buffers will be enough.
                     */

					if (NULL == data->frm_grp)
					{
						/* If buffer group is not set create one and limit it */
						ret = mpp_buffer_group_get_internal(&data->frm_grp, MPP_BUFFER_TYPE_ION);
						if (ret) {
							printf("%p get mpp buffer group failed ret %d\n", ctx, ret);
							break;
						}
                    
						/* Set buffer to mpp decoder */
						ret = mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, data->frm_grp);
						if (ret) {
							printf("%p set buffer group failed ret %d\n", ctx, ret);
							break;
						}
					}
					else
					{
						/* If old buffer group exist clear it */
                        ret = mpp_buffer_group_clear(data->frm_grp);
                        if (ret) {
                            printf("%p clear buffer group failed ret %d\n", ctx, ret);
                            break;
                        }
					}

					/*
                     * All buffer group config done. Set info change ready to let
                     * decoder continue decoding
                     */
                    ret = mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
					if (ret) {
                        printf("%p info change ready failed ret %d\n", ctx, ret);
                        break;
                    }
                } else {
					RK_U32 err_info = mpp_frame_get_errinfo(frame);
					RK_U32 discard = mpp_frame_get_discard(frame);
                    if (err_info || discard) {
                        printf("decoder_get_frame err:%x discard:%x.\n", err_info, discard);
                    }

                    data->frame_count++;
                    printf("decode_get_frame get frame %d\n", data->frame_count);
					
					if (data->fp_output && !err_info){
						// 测试
						dump_mpp_frame_to_file(frame, data->fp_output);

						//cv::Mat rgbImg;
						//YUV420SP2Mat(frame, rgbImg);
						//cv::imwrite("./"+std::to_string(count++)+".jpg", rgbImg);
					}
                }
                frm_eos = mpp_frame_get_eos(frame);
                mpp_frame_deinit(&frame);
                get_frm = 1;
            }

            // try get runtime frame memory usage
            if (data->frm_grp) {
                size_t usage = mpp_buffer_group_usage(data->frm_grp);
                if (usage > data->max_usage)
                    data->max_usage = usage;
            }

            // if last packet is send but last frame is not found continue
            if (pkt_eos && pkt_done && !frm_eos) {
                usleep(10*1000);
                continue;
            }

            if (frm_eos) {
                printf("found last frame\n");
                break;
            }

            if (data->frame_num > 0 && data->frame_count >= data->frame_num) {
                data->eos = 1;
                break;
            }

            if (get_frm)
                continue;
            break;
        } while (1);

        if (data->frame_num > 0 && data->frame_count >= data->frame_num) {
            data->eos = 1;
            printf("reach max frame number %d\n", data->frame_count);
            break;
        }

        if (pkt_done)
            break;

        /*
         * why sleep here:
         * mpi->decode_put_packet will failed when packet in internal queue is
         * full,waiting the package is consumed .Usually hardware decode one
         * frame which resolution is 1080p needs 2 ms,so here we sleep 3ms
         * * is enough.
         */
        usleep(3*1000);
    } while (1);
    mpp_packet_deinit(&packet);

    return ret;
}

//void YUV420P2AVFrame(MppFrame mppFrame)
//{
//	AVFrame *avFrame = av_frame_alloc();

//	avFrame->format = AV_PIX_FMT_YUV420P;
//	avFrame->width = mppFrame.width;
//	avFrame->height = mppFrame.height;

//	int ret = av_frame_get_buffer(avFrame,32);
//	if (ret < 0) 
//	{
//		av_frame_free(avFrame);
//		return;
//	}

//	for (int i = 0;i < mppFrame.height; ++i)
//	{
//		uint8_t *srcY = mppFrame.data[0] + i * mppFrame.linesize[0];
//		uint8_t *dstY = avFrame->data[0] + i * avFrame->linesize[0];

//		memcpy(dstY,srcY,mppFrame->width);
//	}

//	for (int i = 0;i < mppFrame.height/2;++i)
//	{
//		uint8_t *srcU = mppFrame.data[1] + i * mppFrame.linesize[1];
//		uint8_t *dstU = avFrame->data[1] + i * avFrame->linesize[1];

//		uint8_t *srcV = mppFrame.data[2] + i * mppFrame.linesize[2];
//		uint8_t *dstV = avFrame->data[2] + i * avFrame->linesize[2];

//		memcpy(dstU,srcU,mppFrame->width/2);
//		memcpy(dstV,srcV,mppFrame->width/2);
//	}

//	av_frame_free(&avFrame);
//}

void YUV420SP2Mat(MppFrame frame, cv::Mat rgbImg ) {
	RK_U32 width    = 0;
	RK_U32 height   = 0;
	RK_U32 h_stride = 0;
	RK_U32 v_stride = 0;
	MppFrameFormat fmt = MPP_FMT_YUV420SP;
	MppBuffer buffer   = NULL;
	RK_U8 *base = NULL;

	if (NULL == frame)
		return;

	width  = mpp_frame_get_width(frame);
	height = mpp_frame_get_height(frame);
	h_stride = mpp_frame_get_hor_stride(frame);
	v_stride = mpp_frame_get_ver_stride(frame);
	fmt    = mpp_frame_get_fmt(frame);
	buffer = mpp_frame_get_buffer(frame);

	if (NULL == buffer)
		return;

	base = (RK_U8 *)mpp_buffer_get_ptr(buffer);
	
	//RK_U32 buf_size = mpp_frame_get_buf_size(frame);
	//size_t base_length = mpp_buffer_get_size(buffer);
	// printf("base_length = %d\n",base_length);

	// MPP_FMT_YUV420SP || MPP_FMT_YUV420SP_VU
	RK_U32 i;
	RK_U8 *base_y = base;
	RK_U8 *base_c = base + h_stride * v_stride;

	cv::Mat yuvImg;
	yuvImg.create(height * 3 / 2, width, CV_8UC1);

	//转为YUV420sp格式
	int idx = 0;
	for (i = 0; i < height; i++, base_y += h_stride) {
		// fwrite(base_y, 1, width, fp);
		memcpy(yuvImg.data + idx, base_y, width);
		idx += width;
	}
	for (i = 0; i < height / 2; i++, base_c += h_stride) {
		// fwrite(base_c, 1, width, fp);
		memcpy(yuvImg.data + idx, base_c, width);
		idx += width;
	}

	//这里的转码需要转为RGB 3通道， RGBA四通道则不能检测成功
	cv::cvtColor(yuvImg, rgbImg, CV_YUV420sp2RGB);
}

static void fill_MPP_FMT_YUV420SP(RK_U8 *buf, RK_U32 width, RK_U32 height,
                                  RK_U32 hor_stride, RK_U32 ver_stride,
                                  RK_U32 frame_count)
{
    // MPP_FMT_YUV420SP = ffmpeg: nv12
    // https://www.fourcc.org/pixel-format/yuv-nv12/
    RK_U8 *p = buf;
    RK_U32 x, y;

    for (y = 0; y < height; y++, p += hor_stride) {
        for (x = 0; x < width; x++) {
            p[x] = x + y + frame_count * 3;
        }
    }

    p = buf + hor_stride * ver_stride;
    for (y = 0; y < height / 2; y++, p += hor_stride) {
        for (x = 0; x < width / 2; x++) {
            p[x * 2 + 0] = 128 + y + frame_count * 2;
            p[x * 2 + 1] = 64  + x + frame_count * 5;
        }
    }
}

static void fill_MPP_FMT_YUV422SP(RK_U8 *buf, RK_U32 width, RK_U32 height,
                                  RK_U32 hor_stride, RK_U32 ver_stride,
                                  RK_U32 frame_count)
{
    // MPP_FMT_YUV422SP = ffmpeg: nv16
    // not valid in www.fourcc.org
    RK_U8 *p = buf;
    RK_U32 x, y;

    for (y = 0; y < height; y++, p += hor_stride) {
        for (x = 0; x < width; x++) {
            p[x] = x + y + frame_count * 3;
        }
    }

    p = buf + hor_stride * ver_stride;
    for (y = 0; y < height; y++, p += hor_stride) {
        for (x = 0; x < width / 2; x++) {
			p[x * 2 + 0] = 128 + y / 2 + frame_count * 2;
            p[x * 2 + 1] = 64  + x + frame_count * 5;
        }
    }
}

static void get_rgb_color(RK_U32 *R, RK_U32 *G, RK_U32 *B, RK_S32 x, RK_S32 y, RK_S32 frm_cnt)
{
    // frame 0 -> red
    if (frm_cnt == 0) {
        R[0] = 0xff;
        G[0] = 0;
        B[0] = 0;
        return ;
    }

    // frame 1 -> green
    if (frm_cnt == 1) {
        R[0] = 0;
        G[0] = 0xff;
        B[0] = 0;
        return ;
    }

    // frame 2 -> blue
    if (frm_cnt == 2) {
        R[0] = 0;
        G[0] = 0;
        B[0] = 0xff;
        return ;
    }

    // moving color bar
    RK_U8 Y = (0   +  x + y  + frm_cnt * 3);
    RK_U8 U = (128 + (y / 2) + frm_cnt * 2);
    RK_U8 V = (64  + (x / 2) + frm_cnt * 5);

    RK_S32 _R = Y + ((360 * (V - 128)) >> 8);
    RK_S32 _G = Y - (((88 * (U - 128) + 184 * (V - 128))) >> 8);
    RK_S32 _B = Y + ((455 * (U - 128)) >> 8);

    R[0] = MPP_CLIP3(0, 255, _R);
    G[0] = MPP_CLIP3(0, 255, _G);
    B[0] = MPP_CLIP3(0, 255, _B);
}

static void fill_MPP_FMT_RGB565(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_RGB565 = ffmpeg: rgb565be
    // 16 bit pixel     MSB  -------->  LSB
    //                 (rrrr,rggg,gggb,bbbb)
    // big    endian   |  byte 0 |  byte 1 |
    // little endian   |  byte 1 |  byte 0 |
    RK_U16 val = (((R >> 3) & 0x1f) << 11) |
                 (((G >> 2) & 0x3f) <<  5) |
                 (((B >> 3) & 0x1f) <<  0);
    if (be) {
        p[0] = (val >> 8) & 0xff;
        p[1] = (val >> 0) & 0xff;
    } else {
        p[0] = (val >> 0) & 0xff;
        p[1] = (val >> 8) & 0xff;
    }
}

static void fill_MPP_FMT_BGR565(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_BGR565 = ffmpeg: bgr565be
    // 16 bit pixel     MSB  -------->  LSB
    //                 (bbbb,bggg,gggr,rrrr)
    // big    endian   |  byte 0 |  byte 1 |
    // little endian   |  byte 1 |  byte 0 |
    RK_U16 val = (((R >> 3) & 0x1f) <<  0) |
                 (((G >> 2) & 0x3f) <<  5) |
                 (((B >> 3) & 0x1f) << 11);
    if (be) {
        p[0] = (val >> 8) & 0xff;
        p[1] = (val >> 0) & 0xff;
    } else {
        p[0] = (val >> 0) & 0xff;
        p[1] = (val >> 8) & 0xff;
    }
}

static void fill_MPP_FMT_RGB555(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_RGB555 = ffmpeg: rgb555be
    // 16 bit pixel     MSB  -------->  LSB
    //                 (0rrr,rrgg,gggb,bbbb)
    // big    endian   |  byte 0 |  byte 1 |
    // little endian   |  byte 1 |  byte 0 |
    RK_U16 val = (((R >> 3) & 0x1f) << 10) |
                 (((G >> 3) & 0x1f) <<  5) |
                 (((B >> 3) & 0x1f) <<  0);
    if (be) {
        p[0] = (val >> 8) & 0xff;
        p[1] = (val >> 0) & 0xff;
    } else {
        p[0] = (val >> 0) & 0xff;
        p[1] = (val >> 8) & 0xff;
    }
}

static void fill_MPP_FMT_BGR555(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_BGR555 = ffmpeg: bgr555be
    // 16 bit pixel     MSB  -------->  LSB
    //                 (0bbb,bbgg,gggr,rrrr)
    // big    endian   |  byte 0 |  byte 1 |
    // little endian   |  byte 1 |  byte 0 |
    RK_U16 val = (((R >> 3) & 0x1f) <<  0) |
                 (((G >> 3) & 0x1f) <<  5) |
                 (((B >> 3) & 0x1f) << 10);
    if (be) {
        p[0] = (val >> 8) & 0xff;
        p[1] = (val >> 0) & 0xff;
    } else {
        p[0] = (val >> 0) & 0xff;
        p[1] = (val >> 8) & 0xff;
    }
}

static void fill_MPP_FMT_RGB444(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_RGB444 = ffmpeg: rgb444be
    // 16 bit pixel     MSB  -------->  LSB
    //                 (0000,rrrr,gggg,bbbb)
    // big    endian   |  byte 0 |  byte 1 |
    // little endian   |  byte 1 |  byte 0 |
    RK_U16 val = (((R >> 4) & 0xf) << 8) |
                 (((G >> 4) & 0xf) << 4) |
                 (((B >> 4) & 0xf) << 0);
    if (be) {
        p[0] = (val >> 8) & 0xff;
        p[1] = (val >> 0) & 0xff;
    } else {
        p[0] = (val >> 0) & 0xff;
        p[1] = (val >> 8) & 0xff;
    }
}

static void fill_MPP_FMT_BGR444(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_BGR444 = ffmpeg: bgr444be
    // 16 bit pixel     MSB  -------->  LSB
    //                 (0000,bbbb,gggg,rrrr)
    // big    endian   |  byte 0 |  byte 1 |
    // little endian   |  byte 1 |  byte 0 |
    RK_U16 val = (((R >> 4) & 0xf) << 0) |
                 (((G >> 4) & 0xf) << 4) |
                 (((B >> 4) & 0xf) << 8);
    if (be) {
        p[0] = (val >> 8) & 0xff;
        p[1] = (val >> 0) & 0xff;
    } else {
        p[0] = (val >> 0) & 0xff;
        p[1] = (val >> 8) & 0xff;
    }
}

static void fill_MPP_FMT_RGB888(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_RGB888
    // 24 bit pixel     MSB  -------->  LSB
    //                 (rrrr,rrrr,gggg,gggg,bbbb,bbbb)
    // big    endian   |  byte 0 |  byte 1 |  byte 2 |
    // little endian   |  byte 2 |  byte 1 |  byte 0 |
    if (be) {
        p[0] = R;
        p[1] = G;
        p[2] = B;
    } else {
        p[0] = B;
        p[1] = G;
        p[2] = R;
    }
}

static void fill_MPP_FMT_BGR888(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_BGR888
    // 24 bit pixel     MSB  -------->  LSB
    //                 (bbbb,bbbb,gggg,gggg,rrrr,rrrr)
    // big    endian   |  byte 0 |  byte 1 |  byte 2 |
    // little endian   |  byte 2 |  byte 1 |  byte 0 |
    if (be) {
        p[0] = B;
        p[1] = G;
        p[2] = R;
    } else {
        p[0] = R;
        p[1] = G;
        p[2] = B;
    }
}

static void fill_MPP_FMT_RGB101010(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_RGB101010
    // 32 bit pixel     MSB  -------->  LSB
    //                 (00rr,rrrr,rrrr,gggg,gggg,ggbb,bbbb,bbbb)
    // big    endian   |  byte 0 |  byte 1 |  byte 2 |  byte 3 |
    // little endian   |  byte 3 |  byte 2 |  byte 1 |  byte 0 |
    RK_U32 val = (((R * 4) & 0x3ff) << 20) |
                 (((G * 4) & 0x3ff) << 10) |
                 (((B * 4) & 0x3ff) <<  0);
    if (be) {
        p[0] = (val >> 24) & 0xff;
        p[1] = (val >> 16) & 0xff;
        p[2] = (val >>  8) & 0xff;
        p[3] = (val >>  0) & 0xff;
    } else {
        p[0] = (val >>  0) & 0xff;
        p[1] = (val >>  8) & 0xff;
        p[2] = (val >> 16) & 0xff;
        p[3] = (val >> 24) & 0xff;
    }
}

static void fill_MPP_FMT_BGR101010(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_BGR101010
    // 32 bit pixel     MSB  -------->  LSB
    //                 (00bb,bbbb,bbbb,gggg,gggg,ggrr,rrrr,rrrr)
    // big    endian   |  byte 0 |  byte 1 |  byte 2 |  byte 3 |
    // little endian   |  byte 3 |  byte 2 |  byte 1 |  byte 0 |
    RK_U32 val = (((R * 4) & 0x3ff) <<  0) |
                 (((G * 4) & 0x3ff) << 10) |
                 (((B * 4) & 0x3ff) << 20);
    if (be) {
        p[0] = (val >> 24) & 0xff;
        p[1] = (val >> 16) & 0xff;
        p[2] = (val >>  8) & 0xff;
        p[3] = (val >>  0) & 0xff;
    } else {
        p[0] = (val >>  0) & 0xff;
        p[1] = (val >>  8) & 0xff;
        p[2] = (val >> 16) & 0xff;
        p[3] = (val >> 24) & 0xff;
    }
}

static void fill_MPP_FMT_ARGB8888(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_ARGB8888
    // 32 bit pixel     MSB  -------->  LSB
    //                 (XXXX,XXXX,rrrr,rrrr,gggg,gggg,bbbb,bbbb)
    // big    endian   |  byte 0 |  byte 1 |  byte 2 |  byte 3 |
    // little endian   |  byte 3 |  byte 2 |  byte 1 |  byte 0 |
    if (be) {
        p[0] = 0xff;
        p[1] = R;
        p[2] = G;
        p[3] = B;
    } else {
        p[0] = B;
        p[1] = G;
        p[2] = R;
        p[3] = 0xff;
    }
}

static void fill_MPP_FMT_ABGR8888(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_ABGR8888
    // 32 bit pixel     MSB  -------->  LSB
    //                 (XXXX,XXXX,bbbb,bbbb,gggg,gggg,rrrr,rrrr)
    // big    endian   |  byte 0 |  byte 1 |  byte 2 |  byte 3 |
    // little endian   |  byte 3 |  byte 2 |  byte 1 |  byte 0 |
    if (be) {
        p[0] = 0xff;
        p[1] = B;
        p[2] = G;
        p[3] = R;
    } else {
        p[0] = R;
        p[1] = G;
        p[2] = B;
        p[3] = 0xff;
    }
}

static void fill_MPP_FMT_BGRA8888(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_BGRA8888
    // 32 bit pixel     MSB  -------->  LSB
    //                 (bbbb,bbbb,gggg,gggg,rrrr,rrrr,XXXX,XXXX)
    // big    endian   |  byte 0 |  byte 1 |  byte 2 |  byte 3 |
    // little endian   |  byte 3 |  byte 2 |  byte 1 |  byte 0 |
    if (be) {
        p[0] = B;
        p[1] = G;
        p[2] = R;
        p[3] = 0xff;
    } else {
        p[0] = 0xff;
        p[1] = R;
        p[2] = G;
        p[3] = B;
    }
}

static void fill_MPP_FMT_RGBA8888(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be)
{
    // MPP_FMT_RGBA8888
    // 32 bit pixel     MSB  -------->  LSB
    //                 (rrrr,rrrr,gggg,gggg,bbbb,bbbb,XXXX,XXXX)
    // big    endian   |  byte 0 |  byte 1 |  byte 2 |  byte 3 |
    // little endian   |  byte 3 |  byte 2 |  byte 1 |  byte 0 |
    if (be) {
        p[0] = R;
        p[1] = G;
        p[2] = B;
        p[3] = 0xff;
    } else {
        p[0] = 0xff;
        p[1] = B;
        p[2] = G;
        p[3] = R;
    }
}

typedef void (*FillRgbFunc)(RK_U8 *p, RK_U32 R, RK_U32 G, RK_U32 B, RK_U32 be);

FillRgbFunc fill_rgb_funcs[] = {
    fill_MPP_FMT_RGB565,
    fill_MPP_FMT_BGR565,
    fill_MPP_FMT_RGB555,
    fill_MPP_FMT_BGR555,
    fill_MPP_FMT_RGB444,
    fill_MPP_FMT_BGR444,
    fill_MPP_FMT_RGB888,
    fill_MPP_FMT_BGR888,
    fill_MPP_FMT_RGB101010,
    fill_MPP_FMT_BGR101010,
    fill_MPP_FMT_ARGB8888,
    fill_MPP_FMT_ABGR8888,
    fill_MPP_FMT_BGRA8888,
    fill_MPP_FMT_RGBA8888,
};

static RK_S32 util_check_stride_by_pixel(RK_S32 workaround, RK_S32 width,
                                         RK_S32 hor_stride, RK_S32 pixel_size)
{
    if (!workaround && hor_stride < width * pixel_size) {
        printf("warning: stride by bytes %d is smarller than width %d mutiple by pixel size %d\n",
                hor_stride, width, pixel_size);
        printf("multiple stride %d by pixel size %d and set new byte stride to %d\n",
                hor_stride, pixel_size, hor_stride * pixel_size);
        workaround = 1;
    }

    return workaround;
}

static RK_S32 util_check_8_pixel_aligned(RK_S32 workaround, RK_S32 hor_stride,
                                         RK_S32 pixel_aign, RK_S32 pixel_size,
                                         const char *fmt_name)
{
    if (!workaround && hor_stride != MPP_ALIGN(hor_stride, pixel_aign * pixel_size)) {
        printf("warning: vepu only support 8 aligned horizontal stride in pixel for %s with pixel size %d\n",
                fmt_name, pixel_size);
        printf("set byte stride to %d to match the requirement\n",
                MPP_ALIGN(hor_stride, pixel_aign * pixel_size));
        workaround = 1;
    }

    return workaround;
}

MPP_RET fill_image(RK_U8 *buf, RK_U32 width, RK_U32 height,
                   RK_U32 hor_stride, RK_U32 ver_stride, MppFrameFormat fmt,
                   RK_U32 frame_count)
{
    MPP_RET ret = MPP_OK;
    RK_U8 *buf_y = buf;
    RK_U8 *buf_c = buf + hor_stride * ver_stride;
    RK_U32 x, y, i;
    static RK_S32 is_pixel_stride = 0;
    static RK_S32 not_8_pixel = 0;

    switch (fmt & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP : {
        fill_MPP_FMT_YUV420SP(buf, width, height, hor_stride, ver_stride, frame_count);
    } break;
    case MPP_FMT_YUV422SP : {
        fill_MPP_FMT_YUV422SP(buf, width, height, hor_stride, ver_stride, frame_count);
    } break;
    case MPP_FMT_YUV420P : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = x + y + frame_count * 3;
            }
        }

        p = buf_c;
        for (y = 0; y < height / 2; y++, p += hor_stride / 2) {
            for (x = 0; x < width / 2; x++) {
                p[x] = 128 + y + frame_count * 2;
            }
        }

        p = buf_c + hor_stride * ver_stride / 4;
        for (y = 0; y < height / 2; y++, p += hor_stride / 2) {
            for (x = 0; x < width / 2; x++) {
                p[x] = 64 + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV420SP_VU : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = x + y + frame_count * 3;
            }
        }

        p = buf_c;
        for (y = 0; y < height / 2; y++, p += hor_stride) {
            for (x = 0; x < width / 2; x++) {
                p[x * 2 + 1] = 128 + y + frame_count * 2;
                p[x * 2 + 0] = 64  + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV422P : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = x + y + frame_count * 3;
            }
        }

        p = buf_c;
        for (y = 0; y < height; y++, p += hor_stride / 2) {
            for (x = 0; x < width / 2; x++) {
                p[x] = 128 + y / 2 + frame_count * 2;
            }
        }

        p = buf_c + hor_stride * ver_stride / 2;
        for (y = 0; y < height; y++, p += hor_stride / 2) {
            for (x = 0; x < width / 2; x++) {
                p[x] = 64 + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV422SP_VU : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = x + y + frame_count * 3;
            }
        }

        p = buf_c;
        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width / 2; x++) {
                p[x * 2 + 1] = 128 + y / 2 + frame_count * 2;
                p[x * 2 + 0] = 64  + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV422_YUYV : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width / 2; x++) {
                p[x * 4 + 0] = x * 2 + 0 + y + frame_count * 3;
                p[x * 4 + 2] = x * 2 + 1 + y + frame_count * 3;
                p[x * 4 + 1] = 128 + y / 2 + frame_count * 2;
                p[x * 4 + 3] = 64  + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV422_YVYU : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width / 2; x++) {
                p[x * 4 + 0] = x * 2 + 0 + y + frame_count * 3;
                p[x * 4 + 2] = x * 2 + 1 + y + frame_count * 3;
                p[x * 4 + 3] = 128 + y / 2 + frame_count * 2;
                p[x * 4 + 1] = 64  + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV422_UYVY : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width / 2; x++) {
                p[x * 4 + 1] = x * 2 + 0 + y + frame_count * 3;
                p[x * 4 + 3] = x * 2 + 1 + y + frame_count * 3;
                p[x * 4 + 0] = 128 + y / 2 + frame_count * 2;
                p[x * 4 + 2] = 64  + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV422_VYUY : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width / 2; x++) {
                p[x * 4 + 1] = x * 2 + 0 + y + frame_count * 3;
                p[x * 4 + 3] = x * 2 + 1 + y + frame_count * 3;
                p[x * 4 + 2] = 128 + y / 2 + frame_count * 2;
                p[x * 4 + 0] = 64  + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV400 : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = x + y + frame_count * 3;
            }
        }
    } break;
    case MPP_FMT_YUV444SP : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = x + y + frame_count * 3;
            }
        }

        p = buf + hor_stride * ver_stride;
        for (y = 0; y < height; y++, p += hor_stride * 2) {
            for (x = 0; x < width; x++) {
                p[x * 2 + 0] = 128 + y / 2 + frame_count * 2;
                p[x * 2 + 1] = 64  + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_YUV444P : {
        RK_U8 *p = buf_y;

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = x + y + frame_count * 3;
            }
        }
        p = buf + hor_stride * ver_stride;
        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = 128 + y / 2 + frame_count * 2;
            }
        }
        p = buf + hor_stride * ver_stride * 2;
        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0; x < width; x++) {
                p[x] = 64  + x + frame_count * 5;
            }
        }
    } break;
    case MPP_FMT_RGB565 :
    case MPP_FMT_BGR565 :
    case MPP_FMT_RGB555 :
    case MPP_FMT_BGR555 :
    case MPP_FMT_RGB444 :
    case MPP_FMT_BGR444 : {
        RK_U8 *p = buf_y;
        RK_U32 pix_w = 2;
        FillRgbFunc fill = fill_rgb_funcs[fmt - MPP_FRAME_FMT_RGB];

        if (util_check_stride_by_pixel(is_pixel_stride, width, hor_stride, pix_w)) {
            hor_stride *= pix_w;
            is_pixel_stride = 1;
        }

        if (util_check_8_pixel_aligned(not_8_pixel, hor_stride,
                                       8, pix_w, "16bit RGB")) {
            hor_stride = MPP_ALIGN(hor_stride, 16);
            not_8_pixel = 1;
        }

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0, i = 0; x < width; x++, i += pix_w) {
                RK_U32 R, G, B;

                get_rgb_color(&R, &G, &B, x, y, frame_count);
                fill(p + i, R, G, B, MPP_FRAME_FMT_IS_BE(fmt));
            }
        }
    } break;
    case MPP_FMT_RGB101010 :
    case MPP_FMT_BGR101010 :
    case MPP_FMT_ARGB8888 :
    case MPP_FMT_ABGR8888 :
    case MPP_FMT_BGRA8888 :
    case MPP_FMT_RGBA8888 : {
        RK_U8 *p = buf_y;
        RK_U32 pix_w = 4;
        FillRgbFunc fill = fill_rgb_funcs[fmt - MPP_FRAME_FMT_RGB];

        if (util_check_stride_by_pixel(is_pixel_stride, width, hor_stride, pix_w)) {
            hor_stride *= pix_w;
            is_pixel_stride = 1;
        }

        if (util_check_8_pixel_aligned(not_8_pixel, hor_stride,
                                       8, pix_w, "32bit RGB")) {
            hor_stride = MPP_ALIGN(hor_stride, 32);
            not_8_pixel = 1;
        }

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0, i = 0; x < width; x++, i += pix_w) {
                RK_U32 R, G, B;

                get_rgb_color(&R, &G, &B, x, y, frame_count);
                fill(p + i, R, G, B, MPP_FRAME_FMT_IS_BE(fmt));
            }
        }
    } break;
    case MPP_FMT_BGR888 :
    case MPP_FMT_RGB888 : {
        RK_U8 *p = buf_y;
        RK_U32 pix_w = 3;
        FillRgbFunc fill = fill_rgb_funcs[fmt - MPP_FRAME_FMT_RGB];

        if (util_check_stride_by_pixel(is_pixel_stride, width, hor_stride, pix_w)) {
            hor_stride *= pix_w;
            is_pixel_stride = 1;
        }

        if (util_check_8_pixel_aligned(not_8_pixel, hor_stride,
                                       8, pix_w, "24bit RGB")) {
            hor_stride = MPP_ALIGN(hor_stride, 24);
            not_8_pixel = 1;
        }

        for (y = 0; y < height; y++, p += hor_stride) {
            for (x = 0, i = 0; x < width; x++, i += pix_w) {
                RK_U32 R, G, B;

                get_rgb_color(&R, &G, &B, x, y, frame_count);
                fill(p + i, R, G, B, MPP_FRAME_FMT_IS_BE(fmt));
            }
        }
    } break;
    default : {
        printf("filling function do not support type %d\n", fmt);
        ret = MPP_NOK;
    } break;
    }
    return ret;
}

static void get_extension(const char *file_name, char *extension)
{
    size_t length = strlen(file_name);
    size_t ext_len = 0;
    size_t i = 0;
    const char *p = file_name + length - 1;

    while (p >= file_name) {
        if (p[0] == '.') {
            for (i = 0; i < ext_len; i++)
                extension[i] = tolower(p[i + 1]);

            extension[i] = '\0';
            return ;
        }
        ext_len++;
        p--;
    }

    extension[0] = '\0';
}

typedef struct Ext2FrmFmt_t {
    const char      *ext_name;
    MppFrameFormat  format;
} Ext2FrmFmt;

Ext2FrmFmt map_ext_to_frm_fmt[] = {
    {   "yuv420p",              MPP_FMT_YUV420P,                            },
    {   "yuv420sp",             MPP_FMT_YUV420SP,                           },
    {   "yuv422p",              MPP_FMT_YUV422P,                            },
    {   "yuv422sp",             MPP_FMT_YUV422SP,                           },
    {   "yuv422uyvy",           MPP_FMT_YUV422_UYVY,                        },
    {   "yuv422vyuy",           MPP_FMT_YUV422_VYUY,                        },
    {   "yuv422yuyv",           MPP_FMT_YUV422_YUYV,                        },
    {   "yuv422yvyu",           MPP_FMT_YUV422_YVYU,                        },

    {   "abgr8888",             MPP_FMT_ABGR8888,                           },
    {   "argb8888",             MPP_FMT_ARGB8888,                           },
    {   "bgr565",               MPP_FMT_BGR565,                             },
    {   "bgr888",               MPP_FMT_BGR888,                             },
    {   "bgra8888",             MPP_FMT_BGRA8888,                           },
    {   "rgb565",               MPP_FMT_RGB565,                             },
    {   "rgb888",               MPP_FMT_RGB888,                             },
    {   "rgba8888",             MPP_FMT_RGBA8888,                           },

    //{   "fbc",                  MPP_FMT_YUV420SP | MPP_FRAME_FBC_AFBC_V1,   },
};

MPP_RET name_to_frame_format(const char *name, MppFrameFormat *fmt)
{
    RK_U32 i;
    MPP_RET ret = MPP_NOK;
    char ext[50];

    get_extension(name, ext);

    for (i = 0; i < MPP_ARRAY_ELEMS(map_ext_to_frm_fmt); i++) {
        Ext2FrmFmt *info = &map_ext_to_frm_fmt[i];

        if (!strcmp(ext, info->ext_name)) {
            *fmt = info->format;
            ret = MPP_OK;
        }
    }

    return ret;
}

typedef struct Ext2Coding_t {
    const char      *ext_name;
    MppCodingType   coding;
} Ext2Coding;

Ext2Coding map_ext_to_coding[] = {
    {   "h264",             MPP_VIDEO_CodingAVC,    },
    {   "264",              MPP_VIDEO_CodingAVC,    },
    {   "avc",              MPP_VIDEO_CodingAVC,    },

    {   "h265",             MPP_VIDEO_CodingHEVC,   },
    {   "265",              MPP_VIDEO_CodingHEVC,   },
    {   "hevc",             MPP_VIDEO_CodingHEVC,   },

    {   "jpg",              MPP_VIDEO_CodingMJPEG,  },
    {   "jpeg",             MPP_VIDEO_CodingMJPEG,  },
    {   "mjpeg",            MPP_VIDEO_CodingMJPEG,  },
};

MPP_RET name_to_coding_type(const char *name, MppCodingType *coding)
{
    RK_U32 i;
    MPP_RET ret = MPP_NOK;
    char ext[50];

    get_extension(name, ext);

    for (i = 0; i < MPP_ARRAY_ELEMS(map_ext_to_coding); i++) {
        Ext2Coding *info = &map_ext_to_coding[i];

        if (!strcmp(ext, info->ext_name)) {
            *coding = info->coding;
            ret = MPP_OK;
        }
    }

    return ret;
}

