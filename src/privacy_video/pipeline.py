from privacy_video.video import FileVideoSource, VideoWriter


def copy_video(input_path: str, output_path: str) -> None:
    source = FileVideoSource(input_path)
    source.open()

    try:
        info = source.get_info()
        writer = VideoWriter(output_path, info)
        writer.open()

        try:
            while True:
                ok, packet = source.read()
                if not ok or packet is None:
                    break

                writer.write_packet(packet)
        finally:
            writer.close()
    finally:
        source.close()