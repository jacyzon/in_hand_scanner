#ifndef PROJECT_OPENCV_GRABBER_H
#define PROJECT_OPENCV_GRABBER_H

#include <pcl/io/grabber.h>

namespace pcl {
  class OpenNIGrabberCustom : public OpenNIGrabber {

  public:

    OpenNIGrabberCustom() { }

    ~OpenNIGrabberCustom() { }

    template<typename PointT>
    typename pcl::PointCloud<PointT>::Ptr
    convertToXYZRGBPointCloudPub(const boost::shared_ptr<openni_wrapper::Image> &image,
                                 const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image) const {
      return OpenNIGrabber::convertToXYZRGBPointCloud<PointT>(image, depth_image);
    }
  };
}

#endif
