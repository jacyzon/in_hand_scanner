#include <opencv2/imgproc/imgproc.hpp>

class Filter {

public:

  Filter() {
    // when depth value < lowerbound or depth value > upperboud
    // it is considered a hole at that pixel
    lower_bound_ = 10;
    upper_bound_ = 65000;
    closing_size_ = 1;
    blur_kern_size_ = 2;
  }

  /*==============================
  Post processing
  ==============================*/
  void process(cv::Mat &input, cv::Mat &output) {
    cv::Mat &tmp = input;

    // closing
    closing(tmp, tmp, closing_size_);

    // zero block filtering
    zeroBlockFilter(tmp, tmp, 2, lower_bound_, upper_bound_);

    // median blur
    cv::medianBlur(tmp, output, 2 * blur_kern_size_ + 1);
  }

  /*==============================
  Closing

  2*s+1 is the kernel length
  ==============================*/
  void closing(cv::Mat &src, cv::Mat &dist, size_t s) {
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * s + 1, 2 * s + 1), cv::Point(s, s));
    morphologyEx(src, dist, cv::MORPH_CLOSE, element);
  }

  /*==============================
  Zero block filter

  s: kernel length
  lower: min threshold to fill
  upper: max threshold to fill
  ==============================*/
  void zeroBlockFilter(cv::Mat &input, cv::Mat &output, size_t s, ushort lower, ushort upper) {
    for (size_t i = 0; i < output.rows; i++)
      for (size_t j = 0; j < output.cols; j++) {
        if (output.at<ushort>(i, j) < lower || output.at<ushort>(i, j) > upper) {
          ushort max = 0;

          //filter mask
          for (int x = i - s / 2; x <= i + s / 2; x++)
            for (int y = j - s / 2; y <= j + s / 2; y++) {
              if (x >= 0 && x < output.rows && y >= 0 && y < output.cols) {
                ushort tmp = output.at<ushort>(x, y);
                if (tmp < upper && tmp > max) max = tmp;
              }
            }

          output.at<ushort>(i, j) = max;
        }
      }
  }

  int getUpper_bound() const {
    return upper_bound_;
  }

  void setUpper_bound(int upper_bound_) {
    Filter::upper_bound_ = upper_bound_;
  }

  int getLower_bound() const {
    return lower_bound_;
  }

  void setLower_bound(int lower_bound_) {
    Filter::lower_bound_ = lower_bound_;
  }

  int getClosing_size() const {
    return closing_size_;
  }

  void setClosing_size(int closing_size_) {
    Filter::closing_size_ = closing_size_;
  }

  int getBlur_kern_size() const {
    return blur_kern_size_;
  }

  void setBlur_kern_size(int blur_kern_size_) {
    Filter::blur_kern_size_ = blur_kern_size_;
  }

private:
  int upper_bound_;
  int lower_bound_;
  int closing_size_;
  int blur_kern_size_;
};
