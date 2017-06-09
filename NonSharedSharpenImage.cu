#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <sys/time.h>

#define BLUR_SIZE 9
#define USE_2D 0

//define the storage for the blur kernel in GPU Constant Memory
__constant__ float M_d[BLUR_SIZE];

cv::Mat imageRGBA;
cv::Mat imageGrey;
cv::Mat imageLin;
cv::Mat image;
uchar4 *d_rgbaImage__;
uchar4 *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }
const long numPixels = numRows() * numCols();


//
// dtime -
//
// utility routine to return
// the current wall clock time
//
double dtime()
{
        double tseconds = 0.0;
        struct timeval mytime;
        gettimeofday(&mytime,(struct timezone*)0);
        tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
        return( tseconds );
}

//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, uchar4 **greyImage, uchar4 **linImage,
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
        imageRGBA.copyTo(imageLin);
	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}
	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage  = (uchar4 *)imageGrey.ptr<unsigned char>(0);
	*linImage  = (uchar4 *)imageLin.ptr<unsigned char>(0);
	const size_t numPixels = numRows() * numCols();

	//allocate memory on the device for both input and output
	cudaMalloc(d_rgbaImage, numPixels * sizeof(uchar4));
	cudaMalloc(d_greyImage, numPixels * sizeof(uchar4));
	cudaMemset(*d_greyImage, 0, numPixels * sizeof(uchar4)); //make sure no memory is left laying around

	//copy input array to the GPU
	cudaMemcpy(*d_rgbaImage, *inputImage, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice);	

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
	//TODO copy the output back to the host
	const int num_pixels = numRows() * numCols();
	cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, num_pixels * sizeof(uchar4), cudaMemcpyDeviceToHost);	
	
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

__host__ __device__ unsigned char check(int n) {return n > 255 ? 255 : (n < 0 ? 0:n);}
__host__ __device__  int indexBounds(int ndx, int maxNdx) {
   return ndx > (maxNdx - 1) ? (maxNdx - 1) : (ndx < 0 ? 0 : ndx);
}

__host__ __device__ int linearize(int c, int r, int w, int h) {
   return indexBounds(c, w) + indexBounds(r, h)*w;
}

__global__
void conv1D(uchar4* const rgbaImage,uchar4* const greyImage,int numRows, int numCols)
{
	int pix_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int pix_y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (pix_x >= 0 && pix_x < numCols && pix_y >= 0 && pix_y < numRows) { 
		int oneD = linearize(pix_x, pix_y, numCols, numRows);	
		float blurValx = 0;
		float blurValy = 0;
		float blurValz = 0;
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				int imgNdx = linearize(pix_x + j, pix_y + i, numCols, numRows);
				int filterNdx = linearize(1 +j, 1+ i, 3, 3);
				int weight = M_d[filterNdx];
				blurValx += rgbaImage[imgNdx].x * weight;
				blurValy += rgbaImage[imgNdx].y * weight;
				blurValz += rgbaImage[imgNdx].z * weight;
			}
		}
		greyImage[pix_y * numCols + pix_x].x = check((int)blurValx);
		greyImage[pix_y * numCols + pix_x].y = check((int)blurValy);
		greyImage[pix_y * numCols + pix_x].z = check((int)blurValz);
	}
}

// Takes an input image and places the sharpened version in outImage
void linearSharpen(const uchar4 *inImage, uchar4 *outImage,
		size_t numRows, size_t numCols, float *linFilter) {

	for (int pix_y = 0; pix_y < numRows; pix_y++) {
		for (int pix_x = 0; pix_x < numCols; pix_x++) {
			float blurValx = 0;
			float blurValy = 0;
			float blurValz = 0;
			for (int i = -1; i <= 1; ++i) {
				for (int j = -1; j <= 1; ++j) {
					int imgNdx = linearize(pix_x + j, pix_y + i, numCols, numRows);
					int filterNdx = linearize(1 +j, 1+ i, 3, 3);
					int weight = linFilter[filterNdx];
					blurValx += inImage[imgNdx].x * weight;
					blurValy += inImage[imgNdx].y * weight;
					blurValz += inImage[imgNdx].z * weight;
				}
			}
			outImage[pix_y * numCols + pix_x].x = check((int)blurValx);
			outImage[pix_y * numCols + pix_x].y = check((int)blurValy);
			outImage[pix_y * numCols + pix_x].z = check((int)blurValz);
		}
	}
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
							uchar4 * d_rgbaImage,
							uchar4*  d_greyImage,
							size_t numRows,
							size_t numCols)
{
	float M_h[BLUR_SIZE]={-1.0, -1.0, -1.0, -1.0, 9.0, -1.0, -1.0, -1.0, -1.0};  //change this to whatever 1D filter you are using
	cudaMemcpyToSymbol(M_d,M_h, BLUR_SIZE*sizeof(float)); //allocates/copy to Constant Memory on the GPU
	//temp image
	uchar4 *d_greyImageTemp;
	cudaMalloc((void **)&d_greyImageTemp, sizeof(uchar4) * numRows*numCols);
	cudaMemset(d_greyImageTemp, 0, numRows*numCols * sizeof(uchar4)); //make sure no memory is left laying around
	
	int threadSize=16;
	int gridSizeX=(numCols + threadSize - 1)/threadSize; 
	int gridSizeY=(numRows + threadSize - 1)/threadSize;
	const dim3 blockSize(threadSize, threadSize, 1);
	const dim3 gridSize(gridSizeX, gridSizeY, 1);
	//for (int i=0;i<30;i++){
		//row
		conv1D<<<gridSize, blockSize>>>(d_rgbaImage,d_greyImage,numRows,numCols);
		cudaDeviceSynchronize();

	//}

}

int main(int argc, char **argv) {
	cudaDeviceReset();

	uchar4 *h_rgbaImage, *d_rgbaImage;
	uchar4 *h_greyImage, *d_greyImage;
        uchar4 *h_linImage;
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
	preProcess(&h_rgbaImage, &h_greyImage, &h_linImage, &d_rgbaImage, &d_greyImage, input_file);
	//call the students' code
	your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
        cudaDeviceSynchronize();
	cudaGetLastError();
	printf("\n");

	// Now time linear version
	double startTime = dtime();
        float linearFilter[] = {-1.0, -1.0, -1.0, -1.0, 9.0, -1.0, -1.0, -1.0, -1.0};
	linearSharpen(h_rgbaImage, h_linImage, numRows(), numCols(), linearFilter);
	printf("Linear runtime: %lf seconds\n", dtime() - startTime);

	postProcess(output_file); //prints gray image

     cudaThreadExit();
     return 0;

}
