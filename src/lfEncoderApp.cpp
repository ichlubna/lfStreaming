#include "encoder.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    constexpr float DEFAULT_QUALITY{1.0};
    constexpr char DEFAULT_FORMAT[] {"H265"};

    Arguments args(argc, argv);
    std::string path = static_cast<std::string>(args["-i"]);
    float quality = static_cast<float>(args["-q"]);
    std::string format = static_cast<std::string>(args["-f"]);
    std::string outputFile = static_cast<std::string>(args["-o"]);

    std::string helpText{ "Usage:\n"
                          "Example: lfEncoder -i /MyAmazingMachine/thoseImages -q 1.0 -f H265 -o ./coolFile.lf\n"
                          "-i - folder with lf grid images\n"
                          "-- the files should be named as row_column (start with zero) naming such as: 0_0.jpg, 0_1.jpg, ...\n"
                          "-q - quality of the encoded stream\n"
                          "-- higher value mean better quality but larger file\n"
                          "-f - format of the video stream: H265, AV1\n"
                          "-o - output file\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;

    if(path == "" || outputFile == "")
    {
        std::cerr << "No paths specified. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }

    if(format == "")
        format = DEFAULT_FORMAT;
    if(!args["-q"])
        quality = DEFAULT_QUALITY;

    try
    {
        Encoder encoder;
        encoder.encode(path, outputFile, quality, format);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
