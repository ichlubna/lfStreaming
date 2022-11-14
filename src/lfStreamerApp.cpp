#include "decoder.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string path = static_cast<std::string>(args["-i"]);

    std::string helpText{ "Usage:\n"
                          "Example: lfStreamer -i /MyAmazingMachine/LFVideo.lf\n"
                          "-i - encoded LF file\n"
                          "-t - specifies camera trajectory, stores views at the positions and closes the app\n"
                          "     trajectory format in normalized (0-1) LF grid coordinates: col_row,col_row,...\n"
                          "     e.g.: -t 0.0_0.0,0.42_0.5,...\n"
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
        Decoder decoder(path);
        if(!args["-t"])
            decoder.decodeAndPlay();
        else
            decoder.decodeAndStore(args["-t"]);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
