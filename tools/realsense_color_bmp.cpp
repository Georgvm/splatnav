#include <cstdint>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <librealsense2/rs.hpp>

namespace {

void write_u16(std::vector<unsigned char>& out, std::uint16_t value) {
    out.push_back(static_cast<unsigned char>(value & 0xff));
    out.push_back(static_cast<unsigned char>((value >> 8) & 0xff));
}

void write_u32(std::vector<unsigned char>& out, std::uint32_t value) {
    out.push_back(static_cast<unsigned char>(value & 0xff));
    out.push_back(static_cast<unsigned char>((value >> 8) & 0xff));
    out.push_back(static_cast<unsigned char>((value >> 16) & 0xff));
    out.push_back(static_cast<unsigned char>((value >> 24) & 0xff));
}

void write_i32(std::vector<unsigned char>& out, std::int32_t value) {
    write_u32(out, static_cast<std::uint32_t>(value));
}

std::vector<unsigned char> rgb_to_bmp(const unsigned char* rgb, int width, int height) {
    const int row_stride = ((width * 3 + 3) / 4) * 4;
    const std::uint32_t pixel_bytes = static_cast<std::uint32_t>(row_stride * height);
    const std::uint32_t file_bytes = 14 + 40 + pixel_bytes;
    std::vector<unsigned char> bmp;
    bmp.reserve(file_bytes);

    bmp.push_back('B');
    bmp.push_back('M');
    write_u32(bmp, file_bytes);
    write_u16(bmp, 0);
    write_u16(bmp, 0);
    write_u32(bmp, 14 + 40);

    write_u32(bmp, 40);
    write_i32(bmp, width);
    write_i32(bmp, height);
    write_u16(bmp, 1);
    write_u16(bmp, 24);
    write_u32(bmp, 0);
    write_u32(bmp, pixel_bytes);
    write_i32(bmp, 2835);
    write_i32(bmp, 2835);
    write_u32(bmp, 0);
    write_u32(bmp, 0);

    std::vector<unsigned char> row(static_cast<std::size_t>(row_stride), 0);
    for (int y = height - 1; y >= 0; --y) {
        const unsigned char* src = rgb + static_cast<std::size_t>(y) * width * 3;
        for (int x = 0; x < width; ++x) {
            row[static_cast<std::size_t>(x) * 3 + 0] = src[static_cast<std::size_t>(x) * 3 + 2];
            row[static_cast<std::size_t>(x) * 3 + 1] = src[static_cast<std::size_t>(x) * 3 + 1];
            row[static_cast<std::size_t>(x) * 3 + 2] = src[static_cast<std::size_t>(x) * 3 + 0];
        }
        bmp.insert(bmp.end(), row.begin(), row.end());
    }
    return bmp;
}

}  // namespace

int main(int argc, char** argv) {
    std::string output_path;
    std::string latest_path;
    std::string intrinsics_path;
    int width = 848;
    int height = 480;
    int fps = 30;
    int warmup_frames = 15;
    bool stream = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--latest-out" && i + 1 < argc) latest_path = argv[++i];
        else if (arg == "--intrinsics-out" && i + 1 < argc) intrinsics_path = argv[++i];
        else if (arg == "--width" && i + 1 < argc) width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) height = std::stoi(argv[++i]);
        else if (arg == "--fps" && i + 1 < argc) fps = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup_frames = std::stoi(argv[++i]);
        else if (arg == "--stream") stream = true;
        else {
            std::cerr << "usage: realsense_color_bmp [--out path] [--stream] [--latest-out path] [--intrinsics-out path] [--width 848] [--height 480] [--fps 30]\n";
            return 2;
        }
    }

    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGB8, fps);
        auto profile = pipe.start(cfg);
        auto color_profile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        auto intr = color_profile.get_intrinsics();
        std::cerr << "color " << intr.width << "x" << intr.height
                  << " fx=" << intr.fx << " fy=" << intr.fy
                  << " ppx=" << intr.ppx << " ppy=" << intr.ppy << "\n";
        if (!intrinsics_path.empty()) {
            std::ofstream intr_out(intrinsics_path);
            intr_out << "{"
                     << "\"width\":" << intr.width
                     << ",\"height\":" << intr.height
                     << ",\"fx\":" << intr.fx
                     << ",\"fy\":" << intr.fy
                     << ",\"cx\":" << intr.ppx
                     << ",\"cy\":" << intr.ppy
                     << "}\n";
        }

        rs2::frameset frames;
        for (int i = 0; i < warmup_frames; ++i) {
            frames = pipe.wait_for_frames(5000);
        }

        if (stream) {
            while (true) {
                frames = pipe.wait_for_frames(5000);
                auto color = frames.get_color_frame();
                if (!color) continue;
                auto bmp = rgb_to_bmp(
                    reinterpret_cast<const unsigned char*>(color.get_data()),
                    color.get_width(),
                    color.get_height()
                );
                if (!latest_path.empty()) {
                    std::string tmp_path = latest_path + ".tmp";
                    std::ofstream latest(tmp_path, std::ios::binary);
                    latest.write(reinterpret_cast<const char*>(bmp.data()), static_cast<std::streamsize>(bmp.size()));
                    latest.close();
                    std::rename(tmp_path.c_str(), latest_path.c_str());
                }
                std::cout << "--frame\r\n"
                          << "Content-Type: image/bmp\r\n"
                          << "Content-Length: " << bmp.size() << "\r\n\r\n";
                std::cout.write(reinterpret_cast<const char*>(bmp.data()), static_cast<std::streamsize>(bmp.size()));
                std::cout << "\r\n";
                std::cout.flush();
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
            }
        }

        auto color = frames.get_color_frame();
        if (!color) throw std::runtime_error("no color frame returned");

        auto bmp = rgb_to_bmp(
            reinterpret_cast<const unsigned char*>(color.get_data()),
            color.get_width(),
            color.get_height()
        );

        if (!output_path.empty()) {
            std::ofstream out(output_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bmp.data()), static_cast<std::streamsize>(bmp.size()));
        } else {
            std::cout.write(reinterpret_cast<const char*>(bmp.data()), static_cast<std::streamsize>(bmp.size()));
        }
        pipe.stop();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "RealSense RGB capture failed: " << exc.what() << "\n";
        return 1;
    }
}
