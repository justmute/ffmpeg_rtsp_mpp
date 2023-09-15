#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
/*Include ffmpeg header file*/
#include <libavformat/avformat.h>
#ifdef __cplusplus
}
#endif
#include "MppDecode.h"


static double get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv,NULL);

	return tv.tv_sec*1000.0 + tv.tv_usec/1000.0;
}

void deInit(MppPacket *packet, MppFrame *frame, MppCtx ctx, char *buf, MpiDecLoopData data )
{
    if (packet) {
        mpp_packet_deinit(packet);
        packet = NULL;
    }

    if (frame) {
        mpp_frame_deinit(frame);
        frame = NULL;
    }

    if (ctx) {
        mpp_destroy(ctx);
        ctx = NULL;
    }

    //if (buf) {
    //    mpp_free(buf);
    //    buf = NULL;
    //}

    if (data.pkt_grp) {
        mpp_buffer_group_put(data.pkt_grp);
        data.pkt_grp = NULL;
    }

    if (data.frm_grp) {
        mpp_buffer_group_put(data.frm_grp);
        data.frm_grp = NULL;
    }

    if (data.fp_output) {
        fclose(data.fp_output);
        data.fp_output = NULL;
    }

    if (data.fp_input) {
        fclose(data.fp_input);
        data.fp_input = NULL;
    }
}

static MppCodingType rkmpp_get_codingtype(AVCodecID codeid)
{
	switch (codeid)
	{
	case AV_CODEC_ID_H264: return MPP_VIDEO_CodingAVC;
	case AV_CODEC_ID_HEVC: return MPP_VIDEO_CodingHEVC;
	case AV_CODEC_ID_VP8:  return MPP_VIDEO_CodingVP8;
	case AV_CODEC_ID_VP9:  return MPP_VIDEO_CodingVP9;
	default:			   return MPP_VIDEO_CodingUnused;
	}
}

int main()
{
    //char filepath[] = "rtsp://admin:Buzhongyao123@192.168.200.228:554/h264/ch39/sub/av_stream";// rtsp 地址
    //char filepath[] = "rtsp://admin:admin@192.168.0.101:8557/h264";	// rtsp 地址
    char filepath[] = "rtsp://admin:123456@192.168.0.123:554/h264";	// rtsp 地址

    //av_register_all();  //函数在ffmpeg4.0以上版本已经被废弃，所以4.0以下版本就需要注册初始函数
    avformat_network_init();
    
	AVDictionary *options = NULL;
    av_dict_set(&options, "buffer_size", "1024000", 0); //设置缓存大小,1080p可将值跳到最大
    av_dict_set(&options, "rtsp_transport", "tcp", 0); //以tcp的方式打开,
    av_dict_set(&options, "stimeout", "5000000", 0); //设置超时断开链接时间，单位us
    av_dict_set(&options, "max_delay", "500000", 0); //设置最大时延

    AVFormatContext *pFormatCtx = NULL;
    pFormatCtx = avformat_alloc_context(); //用来申请AVFormatContext类型变量并初始化默认参数,申请的空间

    //打开网络流或文件流
    if (avformat_open_input(&pFormatCtx, filepath, NULL, &options) != 0)
    {
        printf("Couldn't open input stream.\n");
        return 0;
    }

    //获取视频文件信息
    if (avformat_find_stream_info(pFormatCtx, NULL)<0)
    {
        printf("Couldn't find stream information.\n");
        return 0;
	}

	av_dump_format(pFormatCtx,0,pFormatCtx->url,0);

    //查找码流中是否有视频流
    int videoindex = -1;
    for (unsigned i = 0; i<pFormatCtx->nb_streams; i++)
	{
        if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            videoindex = i;
            break;
        }
	}

    if (videoindex == -1)
    {
        printf("Didn't find a video stream.\n");
        return 0;
    }

	AVCodecID videoCodecID = pFormatCtx->streams[videoindex]->codecpar->codec_id;
	int videoWidth = pFormatCtx->streams[videoindex]->codecpar->width;
	int videoHeight = pFormatCtx->streams[videoindex]->codecpar->height;
	printf("videoWidth = %d, videoHeight = %d \n",videoWidth,videoHeight);

    AVPacket *av_packet = NULL;
    av_packet = (AVPacket *)av_malloc(sizeof(AVPacket)); // 申请空间，存放的每一帧数据 （h264、h265）

	printf("============ FFmpeg Init OK! ============\n");

    //// 初始化
    MPP_RET ret         = MPP_OK;
    //size_t file_size    = 0;

    // base flow context
    MppCtx ctx          = NULL;
    MppApi *mpi         = NULL;

    // input / output
    MppPacket packet    = NULL;
    MppFrame  frame     = NULL;

    MpiCmd mpi_cmd      = MPP_CMD_BASE;
    MppParam param      = NULL;
    RK_U32 need_split   = 1;
	// MppPollType timeout = 5;

    // paramter for resource malloc
    //RK_U32 width        = 2560;
    //RK_U32 height       = 1440;
    RK_U32 width        = videoWidth;
    RK_U32 height       = videoHeight;
    MppCodingType type  = MPP_VIDEO_CodingUnused;

    // resources
    char *buf           = NULL;
    size_t packet_size  = 8*1024;
    MppBuffer pkt_buf   = NULL;
    MppBuffer frm_buf   = NULL;

	type = rkmpp_get_codingtype(videoCodecID);
    if (type == MPP_VIDEO_CodingUnused)
	{
		printf("Unknown codec type (%d)\n",videoCodecID);
	}

	ret = mpp_check_support_format(MPP_CTX_DEC,type);
	if (MPP_OK != ret)
	{
		printf("Codec type (%d) unsupported by MPP\n",videoCodecID);
	}
    
	MpiDecLoopData data;
    memset(&data, 0, sizeof(data));
    
	data.fp_output = fopen("./tenoutput.yuv", "w+b");
    if (NULL == data.fp_output) {
        printf("failed to open output file %s\n", "tenoutput.yuv");
        deInit(&packet, &frame, ctx, buf, data);
    }

    printf("mpi_dec_test decoder test start w %d h %d type %d\n", width, height, type);

    // decoder demo
    ret = mpp_create(&ctx, &mpi);
    if (MPP_OK != ret) {
        printf("mpp_create failed\n");
        deInit(&packet, &frame, ctx, buf, data);
    }

    // NOTE: decoder split mode need to be set before init
    mpi_cmd = MPP_DEC_SET_PARSER_SPLIT_MODE;
    param = &need_split;
    ret = mpi->control(ctx, mpi_cmd, param);
    if (MPP_OK != ret) {
        printf("mpi->control failed\n");
        deInit(&packet, &frame, ctx, buf, data);
    }

    mpi_cmd = MPP_SET_INPUT_BLOCK;
    param = &need_split;
    ret = mpi->control(ctx, mpi_cmd, param);
    if (MPP_OK != ret) {
        printf("mpi->control failed\n");
        deInit(&packet, &frame, ctx, buf, data);
    }

    ret = mpp_init(ctx, MPP_CTX_DEC, type);
    if (MPP_OK != ret) {
        printf("mpp_init failed\n");
        deInit(&packet, &frame, ctx, buf, data);
    }

    data.ctx            = ctx;
    data.mpi            = mpi;
    data.eos            = 0;
    data.packet_size    = packet_size;
    data.frame          = frame;
    data.frame_count    = 0;
	
	printf("============ Mpp Init OK! ============\n");

    double e_record_time,s_record_time;
	int index = 0;
	//这边可以调整i的大小来改变文件中的视频时间,每 +1 就是一帧数据
    while (1)
    {
        if (av_read_frame(pFormatCtx, av_packet) >= 0)
        {
            if (av_packet->stream_index == videoindex)
            {
				printf("------------\ndata size is: %d\n------------", av_packet->size);
                decode_simple(&data, av_packet);
            
				index += 1;
				if (index % 100 == 0)
				{
					e_record_time = get_current_time();
					double runtime = e_record_time - s_record_time;
					double fps = 100*1000/runtime;
					printf("============> video decode: %f ms,%f fps \n",runtime/100,fps);
					s_record_time = e_record_time;
				}

				if (index > 10)
					break;
			}
            if (av_packet != NULL)
                av_packet_unref(av_packet);
        }
    }    

	fclose(data.fp_output);
    av_free(av_packet);
    avformat_close_input(&pFormatCtx);

    return 0;
}