#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat kuwaharaFilter(const Mat& input, int neighborhoodSize) {
    Mat output = Mat::zeros(input.size(), input.type());

    // Calculate integral images
    Mat integralImage, squaredIntegralImage;
    integral(input, integralImage, CV_32S);
    integral(input.mul(input), squaredIntegralImage, CV_32S);

    int halfSize = neighborhoodSize / 2;

    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            // Calculate sub-windows
            int y0 = max(0, y - halfSize);
            int y1 = min(input.rows - 1, y + halfSize);
            int x0 = max(0, x - halfSize);
            int x1 = min(input.cols - 1, x + halfSize);

            // Calculate the areas and means of the sub-windows
            int area = (y1 - y0 + 1) * (x1 - x0 + 1);
            int sum = integralImage.at<int>(y1, x1) - integralImage.at<int>(y0, x1) -
                      integralImage.at<int>(y1, x0) + integralImage.at<int>(y0, x0);
            double mean = static_cast<double>(sum) / area;

            // Calculate the variances of the sub-windows
            int sumSquared = squaredIntegralImage.at<int>(y1, x1) - squaredIntegralImage.at<int>(y0, x1) -
                             squaredIntegralImage.at<int>(y1, x0) + squaredIntegralImage.at<int>(y0, x0);
            double variance = static_cast<double>(sumSquared) / area - mean * mean;

            // Find the minimum variance
            double minVariance = variance;
            int minIndex = 0;

            for (int i = 1; i < 4; ++i) {
                y0 = y + (i <= 1 ? -halfSize : 0);
                y1 = y + (i <= 1 ? 0 : halfSize);
                x0 = x + (i == 0 || i == 2 ? -halfSize : 0);
                x1 = x + (i == 0 || i == 2 ? 0 : halfSize);

                area = (y1 - y0 + 1) * (x1 - x0 + 1);
                sum = integralImage.at<int>(y1, x1) - integralImage.at<int>(y0, x1) -
                      integralImage.at<int>(y1, x0) + integralImage.at<int>(y0, x0);
                mean = static_cast<double>(sum) / area;

                sumSquared = squaredIntegralImage.at<int>(y1, x1) - squaredIntegralImage.at<int>(y0, x1) -
                             squaredIntegralImage.at<int>(y1, x0) + squaredIntegralImage.at<int>(y0, x0);
                variance = static_cast<double>(sumSquared) / area - mean * mean;

                if (variance < minVariance) {
                    minVariance = variance;
                    minIndex = i;
                }
            }

            // Assign the pixel value to the output based on the sub-window with minimum variance
            output.at<uchar>(y, x) = input.at<uchar>(y + (minIndex <= 1 ? -halfSize : 0),
                                                      x + (minIndex == 0 || minIndex == 2 ? -halfSize : 0));
        }
    }

    return output;
}

int main(int argc, char** argv) {
    // Check if correct number of arguments are provided
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " input_image output_image neighborhood_size" << endl;
        return -1;
    }

    // Read input image
    Mat inputImage = imread(argv[1], IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cerr << "Error: Unable to read input image" << endl;
        return -1;
    }

    // Parse neighborhood size
    int neighborhoodSize = atoi(argv[3]);
    if (neighborhoodSize < 3 || neighborhoodSize % 2 == 0 || neighborhoodSize > 15) {
        cerr << "Error: Neighborhood size should be an odd integer between 3 and 15" << endl;
        return -1;
    }

    // Apply Kuwahara filter
    Mat outputImage = kuwaharaFilter(inputImage, neighborhoodSize);

    // Write output image
    imwrite(argv[2], outputImage);

    cout << "Kuwahara filter applied successfully." << endl;

    return 0;
}
