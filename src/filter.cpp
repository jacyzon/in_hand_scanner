#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#define MAX_DILA_SIZE 11
#define MAX_BLUR_KERN 11

using namespace cv;
using namespace std;

Mat src, src_contrast, dist, drawing;

int closing_size = 1;
int blur_kern_size = 3;
const char *windowName1 = "Output";
const char *windowName2 = "Control";

static void track_callback(int, void *);

static void closing(Mat &src, Mat &dist, size_t s);

static void zeroBlockFilter(Mat &input, Mat &output, size_t s);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Too few arguments\n");
    return -1;
  }

  src = imread(argv[1]);
  if (src.empty()) {
    printf("Image data is empty\n");
    return -1;
  }
  src.convertTo(src, CV_8UC1, 1.0, 0);
  src.convertTo(src_contrast, CV_8UC1, 1.0, 2.0);

  namedWindow("Source", CV_WINDOW_FREERATIO);
  imshow("Source", src_contrast);

  Mat dum = Mat::zeros(1, 1, CV_8U);
  namedWindow(windowName2, CV_WINDOW_FREERATIO);
  imshow(windowName2, dum);

  namedWindow(windowName1, CV_WINDOW_FREERATIO);
  createTrackbar("Dila size: ", windowName2, &closing_size, MAX_DILA_SIZE, track_callback);
  createTrackbar("Blur kern: ", windowName2, &blur_kern_size, MAX_BLUR_KERN, track_callback);

  track_callback(0, 0);

  while (true) {
    int c = waitKey(0);
    if (c == 27) break;
    else if (c == 's') {
      FILE *fp = fopen("output.txt", "w");
      if (!fp) break;

      Mat tmp;

      for (int x = 0; x < tmp.rows; x += 5)
        for (int y = 0; y < tmp.cols; y += 5) {
          if (tmp.at<uchar>(x, y) > 10 || tmp.at<uchar>(x, y) < 230)
            fprintf(fp, "%d, %d, %d\n", y, x, 4 * tmp.at<uchar>(x, y));
        }

      fclose(fp);
      printf("Saved successfully");
    }
  }

  return 0;
}

/*==============================
Trackbar callback
==============================*/
static void track_callback(int, void *) {
  //Closing======================================
  closing(src_contrast, dist, closing_size);

  //Zero block filtering=========================
  zeroBlockFilter(dist, dist, 2);

  //Median blur==================================
  medianBlur(dist, drawing, 2 * blur_kern_size + 1);

  imshow(windowName1, drawing);
}


/*==============================
Closing
==============================*/
static void closing(Mat &src, Mat &dist, size_t s) {
  Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * s + 1, 2 * s + 1), Point(s, s));
  morphologyEx(src, dist, MORPH_CLOSE, element);
}


/*==============================
Zero block filter
==============================*/
static void zeroBlockFilter(Mat &input, Mat &output, size_t s) {
//  cvtColor(input, output, CV_RGB2GRAY);

  for (size_t i = 0; i < output.rows; i++)
    for (size_t j = 0; j < output.cols; j++) {
      if (output.at<uchar>(i, j) < 5 || output.at<uchar>(i, j) > 230) {
        uchar max = 0;

        //filter mask
        for (int x = i - s / 2; x <= i + s / 2; x++)
          for (int y = j - s / 2; y <= j + s / 2; y++) {
            uchar tmp = output.at<uchar>(x, y);
            if (tmp < 230 && tmp > max) max = tmp;
          }

        output.at<uchar>(i, j) = max;
      }
    }

//  cvtColor(output, input, CV_GRAY2RGB);
}


/*==============================
Compute gradient
==============================*/
static void gradient(Mat &input, Mat &output, int scale) {
  int gx, gy, norm;
  Mat tmp;
//  cvtColor(input, tmp, CV_RGB2GRAY);
  output = Mat::zeros(tmp.size(), CV_8U);

  for (int x = 1; x < tmp.rows - 1; x++)
    for (int y = 1; y < tmp.cols - 1; y++) {
      gx = (tmp.at<uchar>(x + 1, y) - tmp.at<uchar>(x - 1, y)) / 2;
      gy = (tmp.at<uchar>(x, y + 1) - tmp.at<uchar>(x, y - 1)) / 2;
      norm = scale * sqrt(gx * gx + gy * gy);
      if (norm > 255) output.at<uchar>(x, y) = 255;
      else output.at<uchar>(x, y) = norm;
    }
}
