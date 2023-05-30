//
// Created by Hao on 2023/5/30.
//
#include <opencv2/opencv.hpp>

int main(){
    cv::Mat img = cv::imread("../src/src.jpeg");
    cv::Mat bg = cv::imread("../output/out.jpeg");
    //imshow("raw", img);//原图
    cv::Mat newImg;
    cv::resize(img, newImg,cv::Size(img.cols/4, img.rows/4));//原图太大，这里resize成一半大小
    cv::imshow("resize", newImg);
    cv::Mat mask;
    cv::inRange(newImg, cv::Scalar(15,16,15),cv::Scalar(190,235,255), mask);//获取二值化图像（目标区域赋值为0 否则为255）
    cv::Mat k = getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)); //获取指定形状和尺寸的结构元素
    cv::dilate(mask,mask,k, cv::Point(2,2)); //膨胀
    cv::erode(mask,mask,k,cv::Point(-1,-1),2); //腐蚀
    k = getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::erode(mask,mask,k,cv::Point(-1,-1),3);
    cv::imshow("mask", mask);

    cv::resize(bg, bg, newImg.size());//resize背景图和目标图大小一致
    cv::Mat dstImg = bg;
    newImg.copyTo(dstImg, mask);//按照mask(0不复制， 1复制) 复制newImg
    cv::imshow("dstShow", dstImg);
}