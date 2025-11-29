#include "iostream"
#include "fstream"
#include "string"

int main(void)
{
    std::string output_path = "bin/out.ppm";
    std::ofstream output(output_path, std::ios::binary);
    output << "P6\n";
    int width = 600;
    int height = 480;
    output << std::to_string(width) << " " << std::to_string(height) << "\n";
    output << "255\n";
 
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            unsigned char r = 255;
            unsigned char g = 255;
            unsigned char b = 0;

            r = int((float(x) / float(width)) * 255);
            g = int((float(y) / float(height)) * 255);
           
            output << r << g << b;
        }
    }

    output.close();
}