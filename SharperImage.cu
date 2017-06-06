#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <numeric>
#include <stdlib.h>


#define BLUR_SIZE 5
#define USE_2D 0

//define the storage for the blur kernel in GPU Constant Memory
__constant__ float M_d[BLUR_SIZE];

cv::Mat imageRGBA;
cv::Mat imageGrey;
cv::Mat image;
uchar4 *d_rgbaImage__;
uchar4 *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }
const long numPixels = numRows() * numCols();


//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, uchar4 **greyImage,
				uchar4 **d_rgbaImage, uchar4 **d_greyImage,
				const std::string &filename) {
	//make sure the context initializes ok
	cudaFree(0);
	//Read Image into an OpenCV Matrix
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageRGBA.copyTo(imageGrey);
	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}
	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage  = (uchar4 *)imageGrey.ptr<unsigned char>(0);
	const size_t numPixels = numRows() * numCols();

	//allocate memory on the device for both input and output
	cudaMalloc((void **) d_rgbaImage, numPixels * sizeof(uchar4));
	cudaMalloc((void **) d_greyImage, numPixels * sizeof(uchar4));

	//copy input array to the GPU
	cudaMemcpy(*d_rgbaImage, *inputImage, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice);	

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
	//TODO copy the output back to the host
	cudaMemcpy(imageGrey.ptr<uchar4>(0), d_greyImage__, numPixels * sizeof(uchar4), cudaMemcpyDeviceToHost);	
	
	cudaDeviceSynchronize();
	//change in color space required by OpenCV	
	cv::cvtColor(imageGrey, imageGrey, CV_BGR2RGBA);
	//output the image to a file
	cv::imwrite(output_file.c_str(), imageGrey);
	//display the output image (will only work if you are on the lab machines)
	cv::imshow ("Output Image", imageGrey);
	cv::waitKey(0);
	////cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);

}

__device__ unsigned char check(int n) {return n > 255 ? 255 : (n < 0 ? 0:n);}
__device__  int indexBounds(int ndx, int maxNdx) {
   return ndx > (maxNdx - 1) ? (maxNdx - 1) : (ndx < 0 ? 0 : ndx);
}

__device__ int linearize(int c, int r, int w, int h) {
   return indexBounds(c, w) + indexBounds(r, h)*w;
}

__global__
void conv1D(const uchar4* const rgbaImage,uchar4* const greyImage,int numRows, int numCols)
{

	int pix_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int pix_y = (blockIdx.y * blockDim.y) + threadIdx.y;
   int 
	
	uchar4 blurVal = {0,0,0,1};
	for (int i = -2; i <= 2; i++) {
		cur_x = pix_x + i;
		if (cur_x >= 0 && cur_x<numCols) {
			blurVal.x += rgbaImage[pix_y * numCols + cur_x].x * M_d[i + 2];
			blurVal.y += rgbaImage[pix_y * numCols + cur_x].y * M_d[i + 2];
			blurVal.z += rgbaImage[pix_y * numCols + cur_x].z * M_d[i + 2];
		}
	}
	greyImage[pix_y * numCols + pix_x] = blurVal;
}


void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
							uchar4 * d_rgbaImage,
							uchar4*  d_greyImage,
							size_t numRows,
							size_t numCols)
{
	float M_h[BLUR_SIZE]={0.0625, 0.25, 0.375, 0.25, 0.0625};  //change this to whatever 1D filter you are using
	cudaMemcpyToSymbol(M_d,M_h, BLUR_SIZE*sizeof(float)); //allocates/copy to Constant Memory on the GPU
	//temp image
	uchar4 *d_greyImageTemp;
	cudaMalloc((void **)&d_greyImageTemp, sizeof(uchar4) * numRows*numCols);
	cudaMemset(d_greyImageTemp, 0, numRows*numCols * sizeof(uchar4)); //make sure no memory is left laying around
	
	int threadSize=16;
	int gridSizeX=(numCols + threadSize-1)/threadSize; 
	int gridSizeY=(numRows + threadSize-1)/threadSize;
	const dim3 blockSize(threadSize, threadSize, 1);
	const dim3 gridSize(gridSizeY, gridSizeX, 1);
	for (int i=0;i<1;i++){
		//row
		conv1D<<<gridSize, blockSize>>>(d_rgbaImage,d_greyImageTemp,numRows,numCols);
		cudaDeviceSynchronize();
		//col
		conv1DCol<<<gridSize, blockSize>>>(d_greyImageTemp,d_greyImage,numRows,numCols);
		cudaDeviceSynchronize();

		//swap
		d_rgbaImage=d_greyImage;
	}

}

int main(int argc, char **argv) {
	cudaDeviceReset();

	uchar4 *h_rgbaImage, *d_rgbaImage;
	uchar4 *h_greyImage, *d_greyImage;
	std::string input_file;
	std::string output_file;

	if (argc == 3) {
		input_file  = std::string(argv[1]);
		output_file = std::string(argv[2]);
	}
	else {
		std::cerr << "Usage: ./hw input_file output_file" << std::endl;
		exit(1);
	}

	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);
	//call the students' code
	your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
    cudaDeviceSynchronize();
	cudaGetLastError();
	printf("\n");
	postProcess(output_file); //prints gray image

     cudaThreadExit();
     return 0;

}

