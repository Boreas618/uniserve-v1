/*
 * The FFmpeg 8.1.2 shim of the media crate.
 *
 * It includes the installed libav* headers and exposes a small, stable C
 * interface of the crate's own: encode one media unit and one audio track,
 * assemble an MP4 from encoded units, decode an audio track to planar float,
 * and inspect an encoded file. Rust declares only these functions, so the
 * crate follows FFmpeg's ABI without mirroring its structs.
 *
 * The options are the serving options read from the PyAV implementation the
 * design replaces (`uniserve/media/codec.py`):
 *
 * | stream | options |
 * | --- | --- |
 * | H.264 | libx264, yuv420p, rgb24 converted with swscale's bilinear filter, 8 threads, preset ultrafast, tune zerolatency |
 * | AAC | aac, stereo, aac_coder=fast, 144 kb/s, 8 threads, planar float input |
 * | container | MP4, one video stream per unit template, one audio stream |
 *
 * Six settings that PyAV leaves at their encoder defaults are set here
 * explicitly, because FFmpeg's AVCodecContext defaults differ from them:
 * `bit_rate = 0` (x264's constant-rate-factor rate control, not the 200 kb/s
 * average bitrate default), `gop_size = -1` and `max_b_frames = -1` (the
 * preset's keyframe interval and B-frame count rather than FFmpeg's 12 and 0),
 * and the muxer's global-header flag. A nonzero failure code leaves a message
 * in a thread-local buffer that `usv_error` returns.
 */

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define USV_OK 0
#define USV_ERROR (-1)

#define IO_BLOCK 32768
#define VIDEO_THREADS 8
#define AUDIO_BIT_RATE 144000
#define AUDIO_FRAME_SAMPLES 1024

static _Thread_local char usv_error_message[512];

const char *usv_error(void) { return usv_error_message; }

static int fail(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(usv_error_message, sizeof(usv_error_message), format, args);
    va_end(args);
    return USV_ERROR;
}

static int fail_averror(const char *context, int code) {
    char reason[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(code, reason, sizeof(reason));
    return fail("%s: %s", context, reason);
}

/* ---- growable byte buffer ------------------------------------------------ */

typedef struct {
    uint8_t *data;
    size_t size;
    size_t capacity;
} Buffer;

static void buffer_free(Buffer *buffer) {
    av_freep(&buffer->data);
    buffer->size = 0;
    buffer->capacity = 0;
}

static int buffer_reserve(Buffer *buffer, size_t extra) {
    size_t wanted = buffer->size + extra;
    if (wanted <= buffer->capacity) {
        return USV_OK;
    }
    size_t capacity = buffer->capacity ? buffer->capacity : 65536;
    while (capacity < wanted) {
        capacity *= 2;
    }
    uint8_t *grown = av_realloc(buffer->data, capacity);
    if (!grown) {
        return fail("out of memory growing a media buffer to %zu bytes", capacity);
    }
    buffer->data = grown;
    buffer->capacity = capacity;
    return USV_OK;
}

static int buffer_append(Buffer *buffer, const uint8_t *data, size_t size) {
    if (buffer_reserve(buffer, size) != USV_OK) {
        return USV_ERROR;
    }
    memcpy(buffer->data + buffer->size, data, size);
    buffer->size += size;
    return USV_OK;
}

void usv_free(void *pointer) { av_free(pointer); }

/* ---- AVIO over memory ---------------------------------------------------- */

static int write_packet(void *opaque, const uint8_t *data, int size) {
    Buffer *buffer = opaque;
    if (size <= 0) {
        return 0;
    }
    if (buffer_append(buffer, data, (size_t)size) != USV_OK) {
        return AVERROR(ENOMEM);
    }
    return size;
}

/* Opens an MP4 output context that writes into `buffer`, whose bytes the
 * caller owns after `finish_output`. */
static int open_output_memory(AVFormatContext **output, Buffer *buffer) {
    AVFormatContext *context = NULL;
    int code = avformat_alloc_output_context2(&context, NULL, "mp4", NULL);
    if (code < 0 || !context) {
        return fail_averror("allocating the MP4 muxer", code);
    }
    uint8_t *io_buffer = av_malloc(IO_BLOCK);
    if (!io_buffer) {
        avformat_free_context(context);
        return fail("out of memory allocating an MP4 I/O buffer");
    }
    AVIOContext *io = avio_alloc_context(io_buffer, IO_BLOCK, 1, buffer, NULL,
                                         write_packet, NULL);
    if (!io) {
        av_free(io_buffer);
        avformat_free_context(context);
        return fail("out of memory allocating the MP4 I/O context");
    }
    context->pb = io;
    context->flags |= AVFMT_FLAG_CUSTOM_IO;
    *output = context;
    return USV_OK;
}

/* Flushes and releases an output context opened by `open_output_memory`. */
static void close_output_memory(AVFormatContext **output) {
    AVFormatContext *context = *output;
    if (!context) {
        return;
    }
    if (context->pb) {
        av_freep(&context->pb->buffer);
        avio_context_free(&context->pb);
    }
    avformat_free_context(context);
    *output = NULL;
}

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t position;
} MemoryReader;

static int read_packet(void *opaque, uint8_t *target, int size) {
    MemoryReader *reader = opaque;
    size_t available = reader->size - reader->position;
    size_t take = (size_t)size < available ? (size_t)size : available;
    if (take == 0) {
        return AVERROR_EOF;
    }
    memcpy(target, reader->data + reader->position, take);
    reader->position += take;
    return (int)take;
}

static int64_t seek_memory(void *opaque, int64_t offset, int whence) {
    MemoryReader *reader = opaque;
    int64_t base = 0;
    switch (whence & ~AVSEEK_FORCE) {
        case SEEK_SET:
            base = 0;
            break;
        case SEEK_CUR:
            base = (int64_t)reader->position;
            break;
        case SEEK_END:
            base = (int64_t)reader->size;
            break;
        case AVSEEK_SIZE:
            return (int64_t)reader->size;
        default:
            return AVERROR(EINVAL);
    }
    int64_t wanted = base + offset;
    if (wanted < 0 || (uint64_t)wanted > reader->size) {
        return AVERROR(EINVAL);
    }
    reader->position = (size_t)wanted;
    return wanted;
}

static void close_input_memory(AVFormatContext **input);

/* Opens an input over `data`; the reader keeps the pointer, so `data` must
 * outlive the context. */
static int open_input_memory(const uint8_t *data, size_t size,
                             AVFormatContext **input, MemoryReader *reader) {
    reader->data = data;
    reader->size = size;
    reader->position = 0;

    uint8_t *io_buffer = av_malloc(IO_BLOCK);
    if (!io_buffer) {
        return fail("out of memory allocating a demuxer I/O buffer");
    }
    AVIOContext *io = avio_alloc_context(io_buffer, IO_BLOCK, 0, reader,
                                         read_packet, NULL, seek_memory);
    if (!io) {
        av_free(io_buffer);
        return fail("out of memory allocating a demuxer I/O context");
    }
    AVFormatContext *context = avformat_alloc_context();
    if (!context) {
        av_freep(&io->buffer);
        avio_context_free(&io);
        return fail("out of memory allocating a demuxer context");
    }
    context->pb = io;
    context->flags |= AVFMT_FLAG_CUSTOM_IO;

    int code = avformat_open_input(&context, NULL, NULL, NULL);
    if (code < 0) {
        /* avformat_open_input frees the context on failure. */
        av_freep(&io->buffer);
        avio_context_free(&io);
        return fail_averror("opening an encoded media stream", code);
    }
    code = avformat_find_stream_info(context, NULL);
    if (code < 0) {
        close_input_memory(&context);
        return fail_averror("reading an encoded media stream's headers", code);
    }
    *input = context;
    return USV_OK;
}

static void close_input_memory(AVFormatContext **input) {
    AVFormatContext *context = *input;
    if (!context) {
        return;
    }
    if (context->pb) {
        av_freep(&context->pb->buffer);
        avio_context_free(&context->pb);
    }
    avformat_close_input(&context);
    *input = NULL;
}

/* ---- codec availability -------------------------------------------------- */

int usv_require_codecs(void) {
    const char *names[] = {"libx264", "aac"};
    for (int index = 0; index < 2; index++) {
        const AVCodec *codec = avcodec_find_encoder_by_name(names[index]);
        if (!codec) {
            return fail("the FFmpeg runtime does not provide the %s encoder",
                        names[index]);
        }
    }
    return USV_OK;
}

/* ---- video units --------------------------------------------------------- */

typedef struct {
    AVCodecContext *context;
    struct SwsContext *scaler;
    const AVCodec *codec;
    AVFrame *frame;
    AVPacket *packet;
    AVFormatContext *output;
    AVStream *stream;
    Buffer buffer;
    int width;
    int height;
    int frame_rate;
} VideoEncoder;

static void video_encoder_free(VideoEncoder *encoder) {
    if (encoder->output) {
        close_output_memory(&encoder->output);
    }
    av_frame_free(&encoder->frame);
    av_packet_free(&encoder->packet);
    if (encoder->scaler) {
        sws_freeContext(encoder->scaler);
        encoder->scaler = NULL;
    }
    avcodec_free_context(&encoder->context);
    buffer_free(&encoder->buffer);
}

/* Creates the muxer's video stream from the encoder's parameters. */
static int video_encoder_stream(VideoEncoder *encoder) {
    AVCodecContext *context = encoder->context;
    AVStream *stream = avformat_new_stream(encoder->output, NULL);
    if (!stream) {
        return fail("creating the MP4 video stream failed");
    }
    stream->time_base = (AVRational){1, encoder->frame_rate};
    context->time_base = stream->time_base;
    int code = avcodec_parameters_from_context(stream->codecpar, context);
    if (code < 0) {
        return fail_averror("writing the video stream's parameters", code);
    }
    encoder->stream = stream;
    return USV_OK;
}

static int video_encoder_open(VideoEncoder *encoder, int width, int height,
                              int frame_rate) {
    if (width <= 0 || height <= 0 || frame_rate <= 0) {
        return fail("video dimensions and frame rate must be positive");
    }
    encoder->width = width;
    encoder->height = height;
    encoder->frame_rate = frame_rate;
    encoder->codec = avcodec_find_encoder_by_name("libx264");
    if (!encoder->codec) {
        return fail("the FFmpeg runtime does not provide the libx264 encoder");
    }
    AVCodecContext *context = avcodec_alloc_context3(encoder->codec);
    if (!context) {
        return fail("out of memory allocating the libx264 encoder");
    }
    encoder->context = context;

    int code = open_output_memory(&encoder->output, &encoder->buffer);
    if (code != USV_OK) {
        return code;
    }

    context->width = width;
    context->height = height;
    context->pix_fmt = AV_PIX_FMT_YUV420P;
    context->time_base = (AVRational){1, frame_rate};
    context->framerate = (AVRational){frame_rate, 1};
    context->thread_count = VIDEO_THREADS;
    /* The serving encoder's rate control and keyframe spacing (libx264.c
     * reads a negative value as "keep the preset's"). */
    context->bit_rate = 0;
    context->gop_size = -1;
    context->max_b_frames = -1;
    if (encoder->output->oformat->flags & AVFMT_GLOBALHEADER) {
        context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    AVDictionary *options = NULL;
    av_dict_set(&options, "preset", "ultrafast", 0);
    av_dict_set(&options, "tune", "zerolatency", 0);
    code = avcodec_open2(context, encoder->codec, &options);
    if (code < 0) {
        av_dict_free(&options);
        return fail_averror("opening libx264", code);
    }
    if (options) {
        av_dict_free(&options);
    }

    encoder->frame = av_frame_alloc();
    encoder->packet = av_packet_alloc();
    if (!encoder->frame || !encoder->packet) {
        return fail("out of memory allocating a video frame");
    }
    encoder->frame->format = AV_PIX_FMT_YUV420P;
    encoder->frame->width = width;
    encoder->frame->height = height;
    code = av_frame_get_buffer(encoder->frame, 0);
    if (code < 0) {
        return fail_averror("allocating a video frame", code);
    }

    encoder->scaler = sws_getContext(width, height, AV_PIX_FMT_RGB24, width,
                                     height, AV_PIX_FMT_YUV420P, SWS_BILINEAR,
                                     NULL, NULL, NULL);
    if (!encoder->scaler) {
        return fail("creating the rgb24 to yuv420p scaler failed");
    }

    if (video_encoder_stream(encoder) != USV_OK) {
        return USV_ERROR;
    }
    code = avformat_write_header(encoder->output, NULL);
    if (code < 0) {
        return fail_averror("writing the MP4 header", code);
    }
    return USV_OK;
}

static int video_encoder_write_packets(VideoEncoder *encoder) {
    for (;;) {
        int code = avcodec_receive_packet(encoder->context, encoder->packet);
        if (code == AVERROR(EAGAIN) || code == AVERROR_EOF) {
            return USV_OK;
        }
        if (code < 0) {
            return fail_averror("encoding a video frame", code);
        }
        encoder->packet->stream_index = encoder->stream->index;
        code = av_interleaved_write_frame(encoder->output, encoder->packet);
        av_packet_unref(encoder->packet);
        if (code < 0) {
            return fail_averror("muxing a video packet", code);
        }
    }
}

static int video_encoder_send(VideoEncoder *encoder, const uint8_t *rgb24,
                              int64_t index) {
    int code = av_frame_make_writable(encoder->frame);
    if (code < 0) {
        return fail_averror("making a video frame writable", code);
    }
    const uint8_t *source[4] = {rgb24, NULL, NULL, NULL};
    int source_stride[4] = {encoder->width * 3, 0, 0, 0};
    sws_scale(encoder->scaler, source, source_stride, 0, encoder->height,
              encoder->frame->data, encoder->frame->linesize);
    encoder->frame->pts = index;
    code = avcodec_send_frame(encoder->context, encoder->frame);
    if (code < 0) {
        return fail_averror("sending a video frame to libx264", code);
    }
    return video_encoder_write_packets(encoder);
}

int usv_encode_video_unit(const uint8_t *rgb24, int frames, int width,
                          int height, int frame_rate, uint8_t **data,
                          size_t *size) {
    if (!rgb24 || frames < 1) {
        return fail("a media unit needs at least one frame");
    }
    VideoEncoder encoder;
    memset(&encoder, 0, sizeof(encoder));
    int code = video_encoder_open(&encoder, width, height, frame_rate);
    if (code != USV_OK) {
        video_encoder_free(&encoder);
        return code;
    }
    size_t frame_bytes = (size_t)width * (size_t)height * 3;
    for (int index = 0; index < frames; index++) {
        if (video_encoder_send(&encoder, rgb24 + (size_t)index * frame_bytes,
                               index) != USV_OK) {
            video_encoder_free(&encoder);
            return USV_ERROR;
        }
    }
    code = avcodec_send_frame(encoder.context, NULL);
    if (code < 0 && code != AVERROR_EOF) {
        video_encoder_free(&encoder);
        return fail_averror("flushing libx264", code);
    }
    if (video_encoder_write_packets(&encoder) != USV_OK) {
        video_encoder_free(&encoder);
        return USV_ERROR;
    }
    code = av_write_trailer(encoder.output);
    if (code < 0) {
        video_encoder_free(&encoder);
        return fail_averror("writing the MP4 trailer", code);
    }
    avio_flush(encoder.output->pb);
    *data = encoder.buffer.data;
    *size = encoder.buffer.size;
    encoder.buffer.data = NULL;
    encoder.buffer.size = 0;
    encoder.buffer.capacity = 0;
    video_encoder_free(&encoder);
    if (*size == 0) {
        return fail("the video encoder wrote no MP4 bytes");
    }
    return USV_OK;
}

/* ---- audio tracks -------------------------------------------------------- */

typedef struct {
    AVCodecContext *context;
    AVFrame *frame;
    AVPacket *packet;
    AVFormatContext *output;
    AVStream *stream;
    Buffer buffer;
    int rate;
} AudioEncoder;

static void audio_encoder_free(AudioEncoder *encoder) {
    if (encoder->output) {
        close_output_memory(&encoder->output);
    }
    av_frame_free(&encoder->frame);
    av_packet_free(&encoder->packet);
    avcodec_free_context(&encoder->context);
    buffer_free(&encoder->buffer);
}

static int audio_encoder_open(AudioEncoder *encoder, int rate) {
    if (rate <= 0) {
        return fail("an audio rate must be positive");
    }
    encoder->rate = rate;
    const AVCodec *codec = avcodec_find_encoder_by_name("aac");
    if (!codec) {
        return fail("the FFmpeg runtime does not provide the AAC encoder");
    }
    AVCodecContext *context = avcodec_alloc_context3(codec);
    if (!context) {
        return fail("out of memory allocating the AAC encoder");
    }
    encoder->context = context;

    int code = open_output_memory(&encoder->output, &encoder->buffer);
    if (code != USV_OK) {
        return code;
    }

    context->sample_fmt = AV_SAMPLE_FMT_FLTP;
    context->sample_rate = rate;
    av_channel_layout_default(&context->ch_layout, 2);
    context->bit_rate = AUDIO_BIT_RATE;
    context->time_base = (AVRational){1, rate};
    context->thread_count = VIDEO_THREADS;
    if (encoder->output->oformat->flags & AVFMT_GLOBALHEADER) {
        context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    AVDictionary *options = NULL;
    av_dict_set(&options, "aac_coder", "fast", 0);
    code = avcodec_open2(context, codec, &options);
    if (code < 0) {
        av_dict_free(&options);
        return fail_averror("opening the AAC encoder", code);
    }
    if (options) {
        av_dict_free(&options);
    }
    if (context->frame_size <= 0) {
        return fail("the AAC encoder declared no frame size");
    }

    AVStream *stream = avformat_new_stream(encoder->output, NULL);
    if (!stream) {
        return fail("creating the MP4 audio stream failed");
    }
    stream->time_base = context->time_base;
    code = avcodec_parameters_from_context(stream->codecpar, context);
    if (code < 0) {
        return fail_averror("writing the audio stream's parameters", code);
    }
    encoder->stream = stream;

    encoder->frame = av_frame_alloc();
    encoder->packet = av_packet_alloc();
    if (!encoder->frame || !encoder->packet) {
        return fail("out of memory allocating an audio frame");
    }
    encoder->frame->format = AV_SAMPLE_FMT_FLTP;
    encoder->frame->sample_rate = rate;
    av_channel_layout_default(&encoder->frame->ch_layout, 2);
    encoder->frame->nb_samples = context->frame_size;
    code = av_frame_get_buffer(encoder->frame, 0);
    if (code < 0) {
        return fail_averror("allocating an audio frame", code);
    }
    code = avformat_write_header(encoder->output, NULL);
    if (code < 0) {
        return fail_averror("writing the AAC MP4 header", code);
    }
    return USV_OK;
}

static int audio_encoder_write_packets(AudioEncoder *encoder) {
    for (;;) {
        int code = avcodec_receive_packet(encoder->context, encoder->packet);
        if (code == AVERROR(EAGAIN) || code == AVERROR_EOF) {
            return USV_OK;
        }
        if (code < 0) {
            return fail_averror("encoding audio", code);
        }
        encoder->packet->stream_index = encoder->stream->index;
        code = av_interleaved_write_frame(encoder->output, encoder->packet);
        av_packet_unref(encoder->packet);
        if (code < 0) {
            return fail_averror("muxing an audio packet", code);
        }
    }
}

/* Sends `samples` interleaved int16 stereo samples at `offset`. The encoder
 * frames hold `frame_size` samples; PyAV's s16p input is converted to the
 * encoder's planar float by the factor 1/32768, which is exact. */
static int audio_encoder_send(AudioEncoder *encoder, const int16_t *pcm,
                              int64_t offset, int samples) {
    int code = av_frame_make_writable(encoder->frame);
    if (code < 0) {
        return fail_averror("making an audio frame writable", code);
    }
    int channels = encoder->frame->ch_layout.nb_channels;
    int frame_samples = encoder->frame->nb_samples;
    for (int channel = 0; channel < channels; channel++) {
        float *plane = (float *)encoder->frame->data[channel];
        for (int sample = 0; sample < frame_samples; sample++) {
            int64_t at = offset + sample;
            if (at < samples) {
                plane[sample] =
                    (float)pcm[at * channels + channel] / 32768.0f;
            } else {
                plane[sample] = 0.0f;
            }
        }
    }
    encoder->frame->pts = offset;
    code = avcodec_send_frame(encoder->context, encoder->frame);
    if (code < 0) {
        return fail_averror("sending an audio frame to the AAC encoder", code);
    }
    return audio_encoder_write_packets(encoder);
}

int usv_encode_audio_track(const int16_t *pcm, int samples, int rate,
                           uint8_t **data, size_t *size) {
    if (samples < 1 || !pcm) {
        return fail("an audio track needs at least one sample");
    }
    AudioEncoder encoder;
    memset(&encoder, 0, sizeof(encoder));
    int code = audio_encoder_open(&encoder, rate);
    if (code != USV_OK) {
        audio_encoder_free(&encoder);
        return code;
    }
    int frame_samples = encoder.context->frame_size;
    for (int64_t offset = 0; offset < samples; offset += frame_samples) {
        if (audio_encoder_send(&encoder, pcm, offset, samples) != USV_OK) {
            audio_encoder_free(&encoder);
            return USV_ERROR;
        }
    }
    code = avcodec_send_frame(encoder.context, NULL);
    if (code < 0 && code != AVERROR_EOF) {
        audio_encoder_free(&encoder);
        return fail_averror("flushing the AAC encoder", code);
    }
    if (audio_encoder_write_packets(&encoder) != USV_OK) {
        audio_encoder_free(&encoder);
        return USV_ERROR;
    }
    code = av_write_trailer(encoder.output);
    if (code < 0) {
        audio_encoder_free(&encoder);
        return fail_averror("writing the AAC MP4 trailer", code);
    }
    avio_flush(encoder.output->pb);
    *data = encoder.buffer.data;
    *size = encoder.buffer.size;
    encoder.buffer.data = NULL;
    encoder.buffer.size = 0;
    encoder.buffer.capacity = 0;
    audio_encoder_free(&encoder);
    if (*size == 0) {
        return fail("the audio encoder wrote no MP4 bytes");
    }
    return USV_OK;
}

/* ---- MP4 assembly -------------------------------------------------------- */

typedef struct UsvMuxer UsvMuxer;

struct UsvMuxer {
    AVFormatContext *output;
    AVStream *video;
    AVStream *audio;
    Buffer buffer;
};

/* Adds an output stream whose parameters match `input`'s stream `index`. */
static int muxer_template_stream(UsvMuxer *muxer, AVFormatContext *input,
                                 int index, AVStream **stream) {
    AVStream *source = input->streams[index];
    AVStream *target = avformat_new_stream(muxer->output, NULL);
    if (!target) {
        return fail("creating an MP4 stream failed");
    }
    target->time_base = source->time_base;
    int code = avcodec_parameters_copy(target->codecpar, source->codecpar);
    if (code < 0) {
        return fail_averror("copying stream parameters", code);
    }
    /* The MP4 muxer writes its own codec tag. */
    target->codecpar->codec_tag = 0;
    target->id = source->id;
    *stream = target;
    return USV_OK;
}

int usv_mux_open(const uint8_t *first_unit, size_t first_size,
                 const uint8_t *audio_template, size_t audio_size,
                 UsvMuxer **muxer_out) {
    UsvMuxer *muxer = calloc(1, sizeof(UsvMuxer));
    if (!muxer) {
        return fail("out of memory allocating an MP4 assembly");
    }
    MemoryReader reader;
    AVFormatContext *unit = NULL;
    AVFormatContext *track = NULL;
    int code = open_output_memory(&muxer->output, &muxer->buffer);
    if (code != USV_OK) {
        goto fail;
    }
    code = open_input_memory(first_unit, first_size, &unit, &reader);
    if (code != USV_OK) {
        goto fail;
    }
    int index = av_find_best_stream(unit, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (index < 0) {
        fail("an encoded media unit has no video");
        goto fail;
    }
    if (muxer_template_stream(muxer, unit, index, &muxer->video) != USV_OK) {
        goto fail;
    }
    close_input_memory(&unit);

    code = open_input_memory(audio_template, audio_size, &track, &reader);
    if (code != USV_OK) {
        goto fail;
    }
    index = av_find_best_stream(track, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    if (index < 0) {
        fail("an encoded audio track has no audio");
        goto fail;
    }
    if (muxer_template_stream(muxer, track, index, &muxer->audio) != USV_OK) {
        goto fail;
    }
    close_input_memory(&track);

    code = avformat_write_header(muxer->output, NULL);
    if (code < 0) {
        fail_averror("writing the assembled MP4's header", code);
        goto fail;
    }
    *muxer_out = muxer;
    return USV_OK;

fail:
    close_input_memory(&unit);
    close_input_memory(&track);
    if (muxer->output) {
        close_output_memory(&muxer->output);
    }
    buffer_free(&muxer->buffer);
    free(muxer);
    return USV_ERROR;
}

/* Copies every packet of `input`'s stream `index` into `target`, shifted by
 * `offset` in the stream's time base. A zero packet duration counts one tick.
 * Returns the decode time after the last packet. */
static int mux_stream(UsvMuxer *muxer, AVFormatContext *input, int index,
                      AVStream *target, int64_t offset, int64_t *last) {
    AVPacket *packet = av_packet_alloc();
    if (!packet) {
        return fail("out of memory allocating a packet");
    }
    int64_t end = offset;
    int code = USV_OK;
    for (;;) {
        int status = av_read_frame(input, packet);
        if (status == AVERROR_EOF) {
            break;
        }
        if (status < 0) {
            fail_averror("reading an encoded stream", status);
            code = USV_ERROR;
            break;
        }
        if (packet->stream_index != index) {
            av_packet_unref(packet);
            continue;
        }
        packet->pts = (packet->pts == AV_NOPTS_VALUE ? 0 : packet->pts) + offset;
        packet->dts = (packet->dts == AV_NOPTS_VALUE ? 0 : packet->dts) + offset;
        packet->stream_index = target->index;
        if (last) {
            int64_t duration = packet->duration ? packet->duration : 1;
            if (packet->dts + duration > end) {
                end = packet->dts + duration;
            }
        }
        status = av_interleaved_write_frame(muxer->output, packet);
        av_packet_unref(packet);
        if (status < 0) {
            fail_averror("muxing an assembled packet", status);
            code = USV_ERROR;
            break;
        }
    }
    av_packet_free(&packet);
    if (last) {
        *last = end;
    }
    return code;
}

int usv_mux_append(UsvMuxer *muxer, const uint8_t *unit, size_t size,
                   int64_t offset, int64_t *next) {
    MemoryReader reader;
    AVFormatContext *input = NULL;
    int code = open_input_memory(unit, size, &input, &reader);
    if (code != USV_OK) {
        return code;
    }
    int index = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (index < 0) {
        close_input_memory(&input);
        return fail("an encoded media unit has no video");
    }
    code = mux_stream(muxer, input, index, muxer->video, offset, next);
    close_input_memory(&input);
    return code;
}

int usv_mux_finish(UsvMuxer *muxer, const uint8_t *audio, size_t size,
                   uint8_t **data, size_t *out_size) {
    MemoryReader reader;
    AVFormatContext *input = NULL;
    int code = open_input_memory(audio, size, &input, &reader);
    if (code != USV_OK) {
        return code;
    }
    int index = av_find_best_stream(input, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    if (index < 0) {
        close_input_memory(&input);
        return fail("an encoded audio track has no audio");
    }
    code = mux_stream(muxer, input, index, muxer->audio, 0, NULL);
    close_input_memory(&input);
    if (code != USV_OK) {
        return code;
    }
    code = av_write_trailer(muxer->output);
    if (code < 0) {
        return fail_averror("writing the assembled MP4's trailer", code);
    }
    avio_flush(muxer->output->pb);
    *data = muxer->buffer.data;
    *out_size = muxer->buffer.size;
    muxer->buffer.data = NULL;
    muxer->buffer.size = 0;
    muxer->buffer.capacity = 0;
    return USV_OK;
}

void usv_mux_close(UsvMuxer *muxer) {
    if (!muxer) {
        return;
    }
    if (muxer->output) {
        close_output_memory(&muxer->output);
    }
    buffer_free(&muxer->buffer);
    free(muxer);
}

/* ---- audio decoding ------------------------------------------------------ */

/* Decodes the first audio stream of `data` to planar float at the stream's
 * native rate and layout, the format conversion PyAV's resampler performs
 * before the reference resamples the track. */
int usv_decode_audio(const uint8_t *data, size_t size, int *rate, int *channels,
                     float **planar, int64_t *samples) {
    MemoryReader reader;
    AVFormatContext *input = NULL;
    int code = open_input_memory(data, size, &input, &reader);
    if (code != USV_OK) {
        return code;
    }
    const AVCodec *decoder = NULL;
    int index =
        av_find_best_stream(input, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0);
    if (index < 0 || !decoder) {
        close_input_memory(&input);
        return fail("an audio condition carries no audio stream");
    }
    AVCodecContext *context = avcodec_alloc_context3(decoder);
    AVFrame *frame = av_frame_alloc();
    AVPacket *packet = av_packet_alloc();
    SwrContext *resampler = NULL;
    Buffer output;
    memset(&output, 0, sizeof(output));
    int channel_count = 0;
    int sample_rate = 0;

    if (!context || !frame || !packet) {
        code = fail("out of memory allocating an audio decoder");
        goto done;
    }
    code = avcodec_parameters_to_context(context, input->streams[index]->codecpar);
    if (code < 0) {
        fail_averror("reading the audio stream's parameters", code);
        goto done;
    }
    code = avcodec_open2(context, decoder, NULL);
    if (code < 0) {
        fail_averror("opening the audio decoder", code);
        goto done;
    }
    if (context->ch_layout.nb_channels < 1) {
        code = fail("an audio stream declares no channels");
        goto done;
    }

    /* The decoder is fed every packet; the resampler converts each decoded
     * frame to the decoder's own rate and layout with planar float samples. */
    for (;;) {
        int status = av_read_frame(input, packet);
        if (status == AVERROR_EOF) {
            break;
        }
        if (status < 0) {
            fail_averror("reading an audio stream", status);
            code = USV_ERROR;
            goto done;
        }
        if (packet->stream_index != index) {
            av_packet_unref(packet);
            continue;
        }
        status = avcodec_send_packet(context, packet);
        av_packet_unref(packet);
        if (status < 0 && status != AVERROR(EAGAIN)) {
            fail_averror("decoding an audio packet", status);
            code = USV_ERROR;
            goto done;
        }
        for (;;) {
            status = avcodec_receive_frame(context, frame);
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
                break;
            }
            if (status < 0) {
                fail_averror("receiving a decoded audio frame", status);
                code = USV_ERROR;
                goto done;
            }
            if (!resampler) {
                AVChannelLayout layout;
                av_channel_layout_copy(&layout, &frame->ch_layout);
                sample_rate = frame->sample_rate;
                channel_count = layout.nb_channels;
                status = swr_alloc_set_opts2(
                    &resampler, &layout, AV_SAMPLE_FMT_FLTP, sample_rate,
                    &layout, frame->format, sample_rate, 0, NULL);
                av_channel_layout_uninit(&layout);
                if (status < 0 || !resampler ||
                    swr_init(resampler) < 0) {
                    fail_averror("creating the audio channel converter", status);
                    code = USV_ERROR;
                    goto done;
                }
            }
            uint8_t **planes = NULL;
            int capacity = swr_get_out_samples(resampler, frame->nb_samples);
            if (capacity <= 0) {
                fail("the audio converter allocated no samples");
                code = USV_ERROR;
                goto done;
            }
            status = av_samples_alloc_array_and_samples(
                &planes, NULL, channel_count, capacity, AV_SAMPLE_FMT_FLTP, 0);
            if (status < 0) {
                fail_averror("allocating converted audio samples", status);
                code = USV_ERROR;
                goto done;
            }
            int converted = swr_convert(resampler, planes, capacity,
                                        (const uint8_t **)frame->extended_data,
                                        frame->nb_samples);
            if (converted < 0) {
                av_freep(&planes[0]);
                av_freep(&planes);
                fail_averror("converting decoded audio", converted);
                code = USV_ERROR;
                goto done;
            }
            for (int channel = 0; channel < channel_count && converted > 0;
                 channel++) {
                if (buffer_append(&output, planes[channel],
                                  (size_t)converted * sizeof(float)) != USV_OK) {
                    av_freep(&planes[0]);
                    av_freep(&planes);
                    code = USV_ERROR;
                    goto done;
                }
            }
            av_freep(&planes[0]);
            av_freep(&planes);
            av_frame_unref(frame);
            if (converted == 0) {
                break;
            }
        }
    }
    /* Flush the decoder's remaining frames. */
    code = avcodec_send_packet(context, NULL);
    (void)code;
    for (;;) {
        int status = avcodec_receive_frame(context, frame);
        if (status == AVERROR_EOF || status == AVERROR(EAGAIN)) {
            break;
        }
        if (status < 0) {
            fail_averror("flushing the audio decoder", status);
            code = USV_ERROR;
            goto done;
        }
        if (!resampler) {
            av_frame_unref(frame);
            continue;
        }
        uint8_t **planes = NULL;
        int capacity = swr_get_out_samples(resampler, frame->nb_samples);
        if (capacity > 0 &&
            av_samples_alloc_array_and_samples(&planes, NULL, channel_count,
                                               capacity, AV_SAMPLE_FMT_FLTP,
                                               0) >= 0) {
            int converted = swr_convert(resampler, planes, capacity,
                                        (const uint8_t **)frame->extended_data,
                                        frame->nb_samples);
            for (int channel = 0; channel < channel_count && converted > 0;
                 channel++) {
                buffer_append(&output, planes[channel],
                              (size_t)converted * sizeof(float));
            }
            av_freep(&planes[0]);
            av_freep(&planes);
        }
        av_frame_unref(frame);
    }
    /* Flush the resampler's delay. */
    if (resampler && output.size > 0) {
        for (;;) {
            uint8_t **planes = NULL;
            int capacity = swr_get_out_samples(resampler, 0);
            if (capacity <= 0) {
                break;
            }
            if (av_samples_alloc_array_and_samples(&planes, NULL, channel_count,
                                                   capacity, AV_SAMPLE_FMT_FLTP,
                                                   0) < 0) {
                break;
            }
            int converted = swr_convert(resampler, planes, capacity, NULL, 0);
            for (int channel = 0; channel < channel_count && converted > 0;
                 channel++) {
                buffer_append(&output, planes[channel],
                              (size_t)converted * sizeof(float));
            }
            av_freep(&planes[0]);
            av_freep(&planes);
            if (converted <= 0) {
                break;
            }
        }
    }

    if (channel_count < 1 || sample_rate <= 0) {
        fail("an audio stream decoded to no samples");
        code = USV_ERROR;
        goto done;
    }
    *rate = sample_rate;
    *channels = channel_count;
    *samples = (int64_t)(output.size / (sizeof(float) * (size_t)channel_count));
    *planar = (float *)output.data;
    output.data = NULL;
    output.size = 0;
    output.capacity = 0;
    code = USV_OK;

done:
    buffer_free(&output);
    if (resampler) {
        swr_free(&resampler);
    }
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&context);
    close_input_memory(&input);
    return code;
}

/* ---- inspection ---------------------------------------------------------- */

typedef struct {
    int32_t has_video;
    int32_t width;
    int32_t height;
    int32_t video_frames;
    int32_t frame_rate_num;
    int32_t frame_rate_den;
    int32_t has_audio;
    int32_t audio_channels;
    int32_t audio_rate;
    int32_t reserved;
    int64_t audio_samples;
    int64_t duration_us;
    int64_t video_duration_us;
} UsvMediaInfo;

static int decode_video_frames(AVFormatContext *input, int index,
                               int64_t *frames, int64_t *duration_us) {
    const AVCodec *decoder = NULL;
    int stream = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, index, -1,
                                     &decoder, 0);
    if (stream < 0 || !decoder) {
        return fail("an inspected file has no video decoder");
    }
    AVCodecContext *context = avcodec_alloc_context3(decoder);
    AVFrame *frame = av_frame_alloc();
    AVPacket *packet = av_packet_alloc();
    if (!context || !frame || !packet) {
        avcodec_free_context(&context);
        av_frame_free(&frame);
        av_packet_free(&packet);
        return fail("out of memory inspecting a video stream");
    }
    int code = avcodec_parameters_to_context(
        context, input->streams[stream]->codecpar);
    if (code >= 0) {
        code = avcodec_open2(context, decoder, NULL);
    }
    if (code < 0) {
        avcodec_free_context(&context);
        av_frame_free(&frame);
        av_packet_free(&packet);
        return fail_averror("opening the video decoder", code);
    }
    int64_t count = 0;
    int64_t last = 0;
    AVRational time_base = input->streams[stream]->time_base;
    for (;;) {
        int status = av_read_frame(input, packet);
        if (status == AVERROR_EOF) {
            break;
        }
        if (status < 0) {
            code = fail_averror("reading the inspected video stream", status);
            goto done;
        }
        if (packet->stream_index != stream) {
            av_packet_unref(packet);
            continue;
        }
        status = avcodec_send_packet(context, packet);
        av_packet_unref(packet);
        if (status < 0 && status != AVERROR(EAGAIN)) {
            continue;
        }
        for (;;) {
            status = avcodec_receive_frame(context, frame);
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
                break;
            }
            if (status < 0) {
                break;
            }
            count++;
            if (frame->pts != AV_NOPTS_VALUE) {
                last = frame->pts;
            }
            av_frame_unref(frame);
        }
    }
    avcodec_send_packet(context, NULL);
    for (;;) {
        int status = avcodec_receive_frame(context, frame);
        if (status < 0) {
            break;
        }
        count++;
        if (frame->pts != AV_NOPTS_VALUE) {
            last = frame->pts;
        }
        av_frame_unref(frame);
    }
    *frames = count;
    *duration_us = av_rescale_q(last, time_base, (AVRational){1, 1000000});
    code = USV_OK;
done:
    avcodec_free_context(&context);
    av_frame_free(&frame);
    av_packet_free(&packet);
    return code;
}

int usv_media_inspect(const uint8_t *data, size_t size, UsvMediaInfo *info) {
    MemoryReader reader;
    AVFormatContext *input = NULL;
    memset(info, 0, sizeof(*info));
    int code = open_input_memory(data, size, &input, &reader);
    if (code != USV_OK) {
        return code;
    }
    info->duration_us = input->duration > 0
                            ? av_rescale(input->duration, 1000000, AV_TIME_BASE)
                            : 0;

    int video = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video >= 0) {
        AVStream *stream = input->streams[video];
        info->has_video = 1;
        info->width = stream->codecpar->width;
        info->height = stream->codecpar->height;
        AVRational rate = stream->avg_frame_rate;
        if (rate.num <= 0 || rate.den <= 0) {
            rate = stream->r_frame_rate;
        }
        info->frame_rate_num = rate.num;
        info->frame_rate_den = rate.den;
        int64_t frames = 0;
        int64_t video_duration_us = 0;
        if (decode_video_frames(input, video, &frames, &video_duration_us) !=
            USV_OK) {
            close_input_memory(&input);
            return USV_ERROR;
        }
        info->video_frames = (int32_t)frames;
        info->video_duration_us = video_duration_us;
    }

    int audio = av_find_best_stream(input, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    if (audio >= 0) {
        AVStream *stream = input->streams[audio];
        info->has_audio = 1;
        info->audio_channels = stream->codecpar->ch_layout.nb_channels;
        info->audio_rate = stream->codecpar->sample_rate;
        info->audio_samples =
            stream->duration > 0
                ? av_rescale_q(stream->duration, stream->time_base,
                               (AVRational){1, info->audio_rate})
                : 0;
    }
    close_input_memory(&input);
    return USV_OK;
}
