#include "decoder.h"
#include "libs/arguments/arguments.hpp"
#include <stdexcept>

void checkPath(std::string path, bool shouldBeDir)
{
    if(!std::filesystem::exists(path))
        throw std::runtime_error("The path " + path + " does not exist!");
    if(shouldBeDir)
    {
        if(!std::filesystem::is_directory(path))
            throw std::runtime_error("The path " + path + " does not lead to a directory!");
    }
    else
    {
        if(std::filesystem::is_directory(path))
            throw std::runtime_error("The path " + path + " does not lead to a file!");
    }
}

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string path = static_cast<std::string>(args["-i"]);

    std::string helpText{ "Usage:\n"
                          "Example: lfStreamer -i /MyAmazingMachine/LFVideo.lf\n"
                          "-i - encoded LF file\n"
                          "-f - starting time frame\n"
                          "-r - playback framerate (unlimited if not specified)\n"
                          "-m - interpolation method - OF_* (optical flow)\n"
                          "     order of 4 closest frames interpolation OF_TB - top-bottom, OF_LF - left-right, OF_D - diagonal\n"
                          "     PP (per pixel)\n"
                          "-t - specifies camera trajectory, stores views at the positions in the start time frame and closes the app\n"
                          "     trajectory format in normalized (0-1) LF grid coordinates: col_row,col_row,...\n"
                          "     e.g.: -t 0.0_0.0,0.42_0.5,...\n"
                          "-o - output directory - used only with -t\n"
                          "Use mouse to change the viewing angle.\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;

    if(!args["-i"])
    {
        std::cerr << "No input specified. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        checkPath(path, false);
        Decoder decoder(path, args["-f"]);
        decoder.setInterpolationMethod(args["-m"]);
        if(!args["-t"])
        {

            float framerate = -1;
            if(!args["-r"])
                framerate = args["-r"];
            decoder.decodeAndPlay(framerate);
        }
        else
        {
            if(!args["-o"])
                throw std::runtime_error("No output path specified.");
            checkPath(args["-o"], true);
            decoder.decodeAndStore(args["-t"], args["-o"]);
        }
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
