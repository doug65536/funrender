#include "funrender.h"
#include "funsdl.h"
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <sys/times.h>
#include "funsdl.h"
#include "perf.h"
#include "objmodel.h"
#include "cpu_usage.h"
#include "abstract_vector.h"
#include "likely.h"

namespace {

duration render_time;
time_point last_avg_time = clk::now();
time_point last_frame_time = clk::now();
uint64_t frame_nr;
int last_frame_nr;
std::array<cpu_usage_info, 2> cpu_usage_samples{[] {
    std::array<cpu_usage_info, 2> buf;
    buf[0] = cpu_usage();
    buf[1] = cpu_usage();
    return buf;
}()};
cpu_usage_info *tms_curr = &cpu_usage_samples[0];
cpu_usage_info *tms_last = &cpu_usage_samples[1];
unsigned user_percent;
unsigned kernel_percent;
unsigned total_percent;

constexpr size_t elap_graph_pts = 1024;
duration elap_graph[elap_graph_pts];
// The data ends before tail, and starts at tail
size_t elap_graph_tail;

#define DEG2RADf(d) ((d)/180.0f*M_PIf)

//stbi_load(pathname, )

}// anonymous namespace end


#include "stb_image.h"

void cleanup(render_ctx * __restrict ctx, int width, int height)
{

}

obj_file_loader::loaded_mesh mesh;

uint32_t *load_image(render_ctx * __restrict ctx,
    int *ret_width, int *ret_height,
    char const *pathname,
    std::string *fail_reason = nullptr)
{
    FILE *pngfile = fopen(pathname, "rb");
    int imgw{};
    int imgh{};
    int comp{};
    uint32_t *pixels = (uint32_t *)stbi_load_from_file(
        pngfile, &imgw, &imgh, &comp, 4);
    fclose(pngfile);
    pngfile = nullptr;
    if (unlikely(!pixels)) {
        if (fail_reason)
            *fail_reason = stbi_failure_reason();
        return nullptr;
    }

    if (!imgw || !imgh || (imgw & (imgw - 1)) || (imgh & (imgh - 1))) {
        if (fail_reason)
            *fail_reason = "Non power of two size unsupported";
        stbi_image_free(pixels);
        return nullptr;
    }

    //flip_colors(pixels, imgw, imgh);
    bind_texture(ctx, 1);

    using size_pair = std::pair<int, int>;

    std::vector<size_pair> sizes;
    size_t total_pixels = 0;
    for (size_pair size{ imgw, imgh }; size.first && size.second;
            (size.first >>= 1), (size.second >>= 1)) {
        total_pixels += size.first * size.second;
        std::cerr << "Adding size " <<
            size.first << "x" << size.second << "\n";
        sizes.push_back(size);
    }

    uint32_t *orig_pixels = pixels;
    total_pixels = (total_pixels + sizeof(vecu32x16) - 1) &
        -sizeof(vecu32x16);
    size_t alloc_sz = sizeof(*pixels) * total_pixels;
    alloc_sz = (alloc_sz + sizeof(vecu32x16) - 1) & -sizeof(vecu32x16);
    pixels = (uint32_t*)aligned_alloc(sizeof(vecu32x16), alloc_sz);
    if (unlikely(!pixels))
        throw std::bad_alloc();

    std::copy(orig_pixels, orig_pixels + imgw * imgh, pixels);
    stbi_image_free(orig_pixels);
    orig_pixels = nullptr;

    uint32_t *input = pixels;
    for (size_pair size : sizes) {
        // Done if at last one
        if (size == sizes.back())
            break;

        uint32_t input_sz = size.first * size.second;
        uint32_t *output = input + input_sz;

        for (size_t y = 0; (int)y < size.second; y += 2) {
            for (size_t x = 0; (int)x < size.first; x += 2) {
                uint32_t inputs[4] = {
                    input[y * size.first + x],  // 🔥 Fixed: Correct indexing
                    input[y * size.first + x + 1],
                    input[(y + 1) * size.first + x],
                    input[(y + 1) * size.first + x + 1]
                };
                int r = (rgb_r(inputs[0]) + rgb_r(inputs[1]) +
                    rgb_r(inputs[2]) + rgb_r(inputs[3])) >> 2;
                int g = (rgb_g(inputs[0]) + rgb_g(inputs[1]) +
                    rgb_g(inputs[2]) + rgb_g(inputs[3])) >> 2;
                int b = (rgb_b(inputs[0]) + rgb_b(inputs[1]) +
                    rgb_b(inputs[2]) + rgb_b(inputs[3])) >> 2;
                uint32_t filtered = rgba(r, g, b, 0xFF);
                output[(y >> 1) * (size.first >> 1) + (x >> 1)] = filtered;  // 🔥 Fixed indexing
            }
        }

        input += input_sz;
        output = input;
    }

    *ret_width = imgw;
    *ret_height = imgh;

    set_texture(ctx, pixels, imgw, imgh, imgw, sizes.size(), free);
    return pixels;
}

int setup(render_ctx * __restrict ctx, int width, int height)
{
    setup_raytrace();

    if (!command_line_files.empty()) {
        obj_file_loader obj;
        obj.load(command_line_files.front().c_str());
        mesh = obj.instantiate();
        //mesh.scale(0.1f);
        mesh.dump_info(std::cerr);
    }

    int orig_width = width;
    int orig_height = height;

    std::string fail_reason;
    uint32_t *pixels = load_image(ctx,
        &width, &height, "earthmap1k.png", &fail_reason);
    if (!pixels) {
        std::cerr << "Texture load failed!\n";
        return EXIT_FAILURE;
    }

    // float fovy = glm::radians(60.0f);
    // float aspect = (float)width / height;
    float znear = 1.0f;
    float zfar = 1048576.0f;

    // Screen aspect ratio
    float aspect_ratio = (float)orig_width / orig_height;

    // Fixed height for normalized NDC
    float fheight = 1.0f;
    // Scale width by aspect
    float fwidth = fheight * aspect_ratio;

    assert(fheight >= 0.0f);

    get_proj_matrix(ctx) = glm::frustum(
        // left, right
        -fwidth, fwidth,
        // bottom, top ✅ Ensure positive fheight
        -fheight, fheight,
        znear, zfar);

    // get_proj_matrix(ctx) = glm::frustum(
    //     -1.0f, 1.0f, -1.0f, 1.0f, znear, zfar);

    get_view_matrix(ctx) = glm::mat4(1.0f);

    srand(44444);
    return EXIT_SUCCESS;
}

// Cube
//     h_____e
//    /|    /|
//  a/_|__d/ |
//  |  |  |  |
//  |  |__|__|
//  | /g  | /f
//  |/____|/
//  b     c
static glm::vec3 cube[] = {
    { -1, -1,  1 }, // a
    { -1,  1,  1 }, // b
    {  1,  1,  1 }, // c
    {  1, -1,  1 }, // d
    {  1, -1, -1 }, // other side a
    {  1,  1, -1 }, // other side b
    { -1,  1, -1 }, // other side c
    { -1, -1, -1 }, // other side d
};

static unsigned int cubeIndices[] = {
    // Front face
    0, 1, 2,
    2, 3, 0,

    // Right face
    3, 2, 4,
    4, 5, 3,

    // Back face
    5, 4, 6,
    6, 7, 5,

    // Left face
    7, 6, 1,
    1, 0, 7,

    // Top face
    1, 6, 5,
    5, 2, 1,

    // Bottom face
    7, 0, 3,
    3, 5, 7
};



void new_render_frame(render_target& __restrict frame,
    render_ctx * __restrict ctx)
{
    parallel_clear(frame, ctx, 0xFF561234);


}

uint64_t handle_input(render_target& __restrict frame,
    render_ctx * __restrict ctx,
    time_point this_time)
{
    duration frame_time = this_time - last_frame_time;
    last_frame_time = this_time;
    elap_graph[elap_graph_tail] = frame_time;
    if (++elap_graph_tail == elap_graph_pts)
        elap_graph_tail = 0;

    uint64_t us_since_last =
        std::chrono::duration_cast<std::chrono::microseconds>(
            frame_time).count();

    float sec_since = us_since_last / 1.0e+6f;

    mouselook_acc.z = 0.0f;
    if (mouselook_pressed[0] && !mouselook_pressed[2])
        mouselook_acc.z = -4.0f;
    else if (mouselook_pressed[2] && !mouselook_pressed[0])
        mouselook_acc.z = 4.0f;
    // else if (mouselook_pressed[2] && mouselook_pressed[0])
    //     mouselook_acc.z *= 0.95f;
    // else
    //     mouselook_vel.z *= 0.91f;

    mouselook_acc.x = 0.0f;
    if (mouselook_pressed[1] && !mouselook_pressed[3])
        mouselook_acc.x = -4.0f;
    else if (mouselook_pressed[3] && !mouselook_pressed[1])
        mouselook_acc.x = 4.0f;
    // else if (mouselook_pressed[3] && mouselook_pressed[1])
    //     mouselook_acc.x *= 0.91f;
    // else
    //     mouselook_vel.x *= 0.91f;

    mouselook_acc.y = 0.0f;
    if (mouselook_pressed[4] && !mouselook_pressed[5])
        mouselook_acc.y = 4.0f;
    else if (mouselook_pressed[5] && !mouselook_pressed[4])
        mouselook_acc.y = -4.0f;
    // else if (mouselook_pressed[5] && mouselook_pressed[4])
    //     mouselook_acc.y = 0.91f;
    // else
    //     mouselook_vel.y *= 0.91f;

    if (!mouselook_pressed[0] &&
        !mouselook_pressed[1] &&
        !mouselook_pressed[2] &&
        !mouselook_pressed[3] &&
        !mouselook_pressed[4] &&
        !mouselook_pressed[5])
        mouselook_vel *= 0.95f;

    glm::vec3 sum =
        mouselook_px * mouselook_acc.x +
        mouselook_py * mouselook_acc.y +
        mouselook_pz * mouselook_acc.z;

    mouselook_vel += sum * sec_since;//= mouselook_acc;
    mouselook_pos += mouselook_vel * sec_since;

    format_text(frame, ctx, -8,
        frame.height - (8+6*24), -1.0f,
        0xFFFFFFFF, "pos=%.3f %.3f %.3f",
        mouselook_pos.x, mouselook_pos.y, mouselook_pos.z);

    format_text(frame, ctx, -8,
        frame.height - (8+4*24), -1.0f,
        0xFFFFFFFF, "pitch=%f, yaw=%f ",
        mouselook_pitch * 180.0f / M_PIf,
        mouselook_yaw * 180.0f / M_PIf);

    format_text(frame, ctx, -8,
        frame.height - (8+3*24), -1.0f,
        0xFFFFFFFF, "x.x=%f, x.y=%f, x.z=%f ",
        mouselook_px.x, mouselook_px.y, mouselook_px.z);

    format_text(frame, ctx, -8,
        frame.height - (8+2*24), -1.0f,
        0xFFFFFFFF, "y.x=%f, y.y=%f, y.z=%f ",
        mouselook_py.x, mouselook_py.y, mouselook_py.z);

    format_text(frame, ctx, -8,
        frame.height - (8+1*24), -1.0f,
        0xFFFFFFFF, "z.x=%f, z.y=%f, z.z=%f ",
        mouselook_pz.x, mouselook_pz.y, mouselook_pz.z);

    return us_since_last;
}

void user_frame(render_target& __restrict frame,
    render_ctx * __restrict ctx)
{
    time_point this_time = clk::now();

    ++frame_nr;

    parallel_clear(frame, ctx, 0xFF101010);// 0xFF561234);

    glm::mat4 view;

    view = glm::mat4(1.0f);
    view = glm::rotate(view, -mouselook_pitch,
        glm::vec3(1.0f, 0.0f, 0.0f));
    view = glm::rotate(view, -mouselook_yaw,
        glm::vec3(0.0f, 1.0f, 0.0f));

    if (raytrace)
        test_raytrace(frame);


    view = glm::translate(view, -mouselook_pos);

    set_transform(ctx, &view, nullptr);

    set_light_pos(ctx, 0, glm::vec4(mouselook_pos, 1.0f));
    set_light_diffuse(ctx, 0, {0.1f, 0.1f, 0.1f});
    set_light_specular(ctx, 0, {0.8f, 0.8f, 0.8f}, 1.0f);
    set_light_enable(ctx, 0, true);

    format_text(frame, ctx,
        -20, 80, -1.0f, 0xff562233, "Test");

    uint64_t us_since_last = handle_input(
        frame, ctx, this_time);

    if (!mesh.vertices.empty()) {
        // time_point huge_st = clk::now();
        draw_elements(frame, ctx,
            mesh.vertices.data(), mesh.vertices.size(),
            mesh.elements.data(), mesh.elements.size());
        // time_point huge_en = clk::now();
        // size_t element_count = mesh.elements.size() / 3;
        // size_t ns_elap = std::chrono::duration_cast<
        //     std::chrono::nanoseconds>(
        //         huge_en - huge_st).count();
        // std::cerr << std::setprecision(3) <<
        //     (element_count * 1.0e+3f / ns_elap) <<
        //     " million elements per second\n";
    }

    size_t constexpr triangle_cnt = 256;
    static float ang[triangle_cnt];
    static float xofs[triangle_cnt];
    static float yofs[triangle_cnt];
    static float zofs[triangle_cnt];
    static float rad[triangle_cnt];
    static float spd[triangle_cnt];
    static float spacing[triangle_cnt];
    //static unsigned col[triangle_cnt];
    static bool init_done;

    // print_matrix("view", std::cerr, view_mtx_stk.back());

    if (!init_done) {
        for (size_t i = 0; i < triangle_cnt; ++i) {
            ang[i] = M_PI_2f;// M_PIf * (float)rand() / (RAND_MAX+1LL);
            if (triangle_cnt == 1) {
                xofs[i] = 0.0f;
                yofs[i] = 0.0f;
                zofs[i] = triangle_cnt - i - 1;
                rad[i] = 0.5f;
            } else {
                xofs[i] = ((float)rand() / (RAND_MAX+1LL) - 0.5f) *
                    frame.width * 0.95f;
                yofs[i] = ((float)rand() / (RAND_MAX+1LL) - 0.5f) *
                    frame.height * 0.95f;
                //zofs[i] = (float)rand() / RAND_MAX * 19.0f + 1.0f;
                zofs[i] = i * 8.0f;
                rad[i] = std::pow(2.0f, (float)rand() /
                    (RAND_MAX+1LL) * 5.9f + 4.0f);
            }

            //col[i] = (float)rand() / RAND_MAX * 0x1000000;
            spd[i] = ((float)rand() / (RAND_MAX+1LL) - 0.5f) * 3.0f;
            spacing[i] = DEG2RADf(90.0f);
            // (float)rand() / RAND_MAX * DEG2RADf(140) + DEG2RADf(40);
        }
        init_done = true;
    }

    duration elap = this_time - last_avg_time;
    if (elap >= std::chrono::seconds(2)) {
        // CPU usage
        *tms_curr = cpu_usage();

        uint64_t tms_total = tms_curr->total - tms_last->total;
        uint64_t tms_user = tms_curr->user - tms_last->user;
        uint64_t tms_kernel = tms_curr->kernel - tms_last->kernel;
        user_percent = 100 * tms_user / tms_total;
        kernel_percent = 100 * tms_kernel / tms_total;
        total_percent = user_percent + kernel_percent;

        std::cerr << "user " << user_percent << "%"
            ", kernel " << kernel_percent << "%"
            ", total " << total_percent << "%\n";
        std::swap(tms_curr, tms_last);

        last_avg_time = this_time;
        auto us = std::chrono::duration_cast<
            std::chrono::microseconds>(elap).count();
        size_t frame_cnt = frame_nr - last_frame_nr;
        size_t fps = frame_cnt * UINT64_C(1000000) / us;
        last_frame_nr = frame_nr;

        duration min_elap = duration::max();
        duration max_elap = duration::min();
        for (size_t i = 0, n = elap_graph_tail; i < elap_graph_pts;
                ++i, (n = ((n + 1) < elap_graph_pts ? n + 1 : 0))) {
            min_elap = std::min(min_elap, elap_graph[n]);
            max_elap = std::max(max_elap, elap_graph[n]);
        }
        size_t diff = std::chrono::duration_cast<std::chrono::microseconds>(
            max_elap - min_elap).count();

        uint64_t render_us = std::chrono::duration_cast<
            std::chrono::microseconds>(render_time).count() / frame_cnt;
        render_time = std::chrono::microseconds(0);

        std::cerr << (us / frame_cnt) << "µs/frame ±" << diff << " µs"
            " (" << fps << " fps) avg (" << render_us << " render µs)\n";

        draw_stats const *stats = get_draw_stats(ctx);

        std::cerr << "clipped=" << stats->clipped <<
            ", unclipped=" << stats->unclipped <<
            ", culled=" << stats->culled <<
            "\n";

        print_vector(std::cerr, mouselook_pos) << ' ' <<
            mouselook_pitch << ' ' << mouselook_yaw << '\n';

        std::array<float, 2> edge_load_factors = get_edge_stats();
        std::cerr << "Edge table load factors: " <<
            edge_load_factors[0] << ", " <<
            edge_load_factors[1] << '\n';
    }

    static size_t const cpu_count = std::thread::hardware_concurrency();

    format_text(frame, ctx, -16,
        frame.height - (64+5*24), -1.0f,
        0xFFFFFFFF, "user %u%% (%zu%%), kernel %u%% (%zu%%), total %u%% (%zu%%)",
        user_percent, user_percent / cpu_count,
        kernel_percent, kernel_percent / cpu_count,
        total_percent, total_percent / cpu_count);

    for (size_t i = 0; i < triangle_cnt; ++i) {
        float ang2 = ang[i] + spacing[i];
        float ang3 = ang[i] + spacing[i] * 1.5;
        float sa1 = std::sin(ang[i]);
        float ca1 = std::cos(ang[i]);
        float sa2 = std::sin(ang2);
        float ca2 = std::cos(ang2);
        float sa3 = std::sin(ang3);
        float ca3 = std::cos(ang3);
        ang[i] += spd[i] * us_since_last / 1000000.0f;
        while (ang[i] >= M_PIf)
            ang[i] -= M_PIf * 2.0f;
        while (ang[i] < -M_PIf)
            ang[i] += M_PIf * 2.0f;
        float sz = rad[i];
        float xo = xofs[i];
        float yo = yofs[i];
        glm::vec4 v[3] = {
            { ca1 * sz + sz + xo, sa1 * sz + sz + yo, zofs[i], 1.0f },
            { ca2 * sz + sz + xo, sa2 * sz + sz + yo, zofs[i], 1.0f },
            { ca3 * sz + sz + xo, sa3 * sz + sz + yo, zofs[i], 1.0f }
        };

        // glm::mat4 const& vm = view_mtx_stk.back();
        // glm::mat4 const& pm = proj_mtx_stk.back();

        // std::cerr << "In:\n";
        // print_vector(std::cerr, v1) << '\n';
        // print_vector(std::cerr, v2) << '\n';
        // print_vector(std::cerr, v3) << '\n';

        glm::vec2 t1{0,0};
        glm::vec3 n1{};
        //glm::vec3 c1{1.0f, 0.0f, 0.0f};
        glm::vec3 c1{1.0f};

        glm::vec2 t2{1,1};
        glm::vec3 n2{};
        glm::vec3 c2{1.0f};

        glm::vec2 t3{1,0};
        glm::vec3 n3{};
        glm::vec3 c3{1.0f};

        n1 = n2 = n3 = glm::cross(
            glm::normalize(glm::vec3(v[1] - v[0])),
            glm::normalize(glm::vec3(v[2] - v[0])));

        draw_polygon(frame, ctx, std::array<scaninfo, 3>{
            scaninfo{t1, v[0], n1, c1},
            scaninfo{t2, v[1], n2, c2},
            scaninfo{t3, v[2], n3, c3}
        });
    }

    fill_box(frame, ctx,
        8, 8, frame.width - 8, 168, -0.99f, 0x80563412, 12);

    if (text_test) {
        draw_text(frame, ctx,
            30, 30+24*0, -1.0f,
            u8"The quick brown fox jumped over the lazy dog 1234567890-=`<>?;'{}\\");
        draw_text(frame, ctx,
            30, 30+24*1, -1.0f,
            u8"THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG !@#$%^&*()_+~,./:\"[]|");
    }
    format_text(frame, ctx, 30, 30+24*2, -1.0f, 0xffffffff,
        "%4" PRIu64 "", smooth_fps / 100);

    format_text(frame, ctx, 30, 30+24*3, -1.0f, 0xffffffff,
        "%4" PRIu64 "", last_fps);
    // draw_text(frame, 10, 10+24*3,
    //     u8"Hello, 你好, こんにちは, שָׁלוֹם, नमस्ते, Γειά σας, Здравствуйте");

    if (text_test) {
        draw_text(frame, ctx, 30, 30+24*4, -1.0f,
            (char const *)u8"a\u0300e\u0301i\u0302o\u0303u\u0308");

        // draw_text(frame, ctx, 30, 30+24*5, -1.0f,
        //     "ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļ"
        //     "ĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻ"
        //     "żŽžſƀƁƂƃƄƅƆƇƈƉƊƋƌƍƎƏƐƑƒƓƔƕƖƗƘƙƚƛƜƝƞƟƠơƢƣƤƥƦƧƨƩƪƫƬƭƮƯưƱƲƳƴƵƶƷƸƹƺ"
        //     "ƻƼƽƾƿǀǁǂǃǄǅǆǇǈǉǊǋǌǍǎǏǐǑǒǓǔǕǖǗǘǙǚǛǜǝǞǟǠǡǢǣǤǥǦǧǨǩǪǫǬǭǮǯǰǱ"
        //     "ǲǳǴǵǶǷǸǹǺǻǼǽǾǿ", nullptr, 64);

        char const *hello_messages[] = {
            u8"English: Hello, World!",
            u8"Spanish: ¡Hola, Mundo!",
            u8"French: Bonjour, Monde!",
            u8"German: Hallo, Welt!",
            u8"Italian: Ciao, Mondo!",
            u8"Portuguese: Olá, Mundo!",
            u8"Dutch: Hallo, Wereld!",
            u8"Swedish: Hej, Världen!",
            u8"Finnish: Hei, Maailma!",
            u8"Danish: Hej, Verden!",
            u8"Polish: Cześć, Świecie!",
            u8"Czech: Ahoj, světe!",
            u8"Russian: Привет, Мир!",
            u8"Greek: Γειά σου Κόσμε!",
            u8"Turkish: Merhaba, Dünya!",
            u8"Arabic: مرحبا، العالم!",
            u8"Hebrew: שלום, עולם!",
            u8"Japanese: こんにちは、世界！",
            u8"Korean: 안녕하세요, 세계!",
            u8"Chinese (Simplified): 你好，世界！",
            u8"Chinese (Traditional): 你好，世界！"
        };

        size_t msgx = 120;
        size_t y = 30+24*2;
        size_t message_count = 0;
        for (char const *message : hello_messages) {
            draw_text(frame, ctx, msgx, y, -1.0f, message);
            y += 24;
            if (++message_count == 10) {
                y -= 24 * message_count;
                msgx += 400;
                message_count = 0;
            }
        }
    }


    render_time += clk::now() - this_time;
}

// 012, 213, 234
// even: n+1, n, n+2
//  odd: n, n+1, n+2
// void texture_strip(render_target & __restrict fp, glm::vec3 *verts,
//     size_t count, size_t stride = 2)
// {
//     for (size_t i = 0; i + 2 <= count; i += 2) {

//     }
// }

// void texture_fan(render_target & __restrict fp, glm::vec3 *verts,
//     size_t count, size_t stride = 2)
// {
//     for (size_t i = 1; i < count; i += stride)
//         texture_raw_triangle(fp, verts[0], verts[i], verts[i + 1]);
// }

// void texture_tris(render_target & __restrict fp, glm::vec3 *verts,
//     size_t count, size_t stride = 3)
// {
//     for (size_t i = 0; i < count; i += stride) {
//         glm::vec4 v4[3] = {
//             glm::vec4(verts[i], 1.0f) * view_mtx_stk.back() * proj_mtx_stk.back(),
//             glm::vec4(verts[i + 1], 1.0f) * view_mtx_stk.back() * proj_mtx_stk.back(),
//             glm::vec4(verts[i + 2], 1.0f) * view_mtx_stk.back() * proj_mtx_stk.back()
//         };
//         for (glm::vec4 &v : v4) {
//             float m = 1.0f / v.w;
//             v.x *= m;
//             v.y *= m;
//             v.z *= m;
//         }
//         texture_raw_triangle(fp, glm::vec3(v4[0]),
//             glm::vec3(v4[1]), glm::vec3(v4[2]));
//     }
// }
