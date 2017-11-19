#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

class Filtros {
private :
	static void thinning(cv::Mat&);
	static void thinningIteration(cv::Mat&, int);
public:
	static void filtroCompleto(cv::Mat&, cv::Mat&);
};

void Filtros::filtroCompleto(cv::Mat& img, cv::Mat& ans) {
	short kernel_gradient_data[] = { 0, -1, 0 ,-1, 5, -1, 0, -1, 0 };
	cv::Mat kernel_gradient(3, 3, CV_8S, kernel_gradient_data);

	short kernel_morph_data[] = { 0, 1, 0 , 1, 1, 1, 0, 1, 0 };
	cv::Mat  kernel_morph(3, 3, CV_8UC1, kernel_morph_data);

	cv::Mat gradientes, umbralizacion, cerradura, apertura;

	cv::filter2D(img, gradientes, -1, kernel_gradient);
	cv::threshold(gradientes, umbralizacion, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::morphologyEx(umbralizacion, cerradura, cv::MORPH_OPEN, kernel_morph);

	cv::bitwise_not(cerradura, ans);

	Filtros::thinning(ans);
}

void Filtros::thinningIteration(cv::Mat& im, int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

	omp_set_num_threads(4);
#pragma omp parallel for
	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)

				marker.at<uchar>(i, j) = 1;
		}
	}
	im &= ~marker;
}

void Filtros::thinning(cv::Mat& im)
{
	// Enforce the range tob e in between 0 - 255
	im /= 255;

	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (countNonZero(diff) > 0);

	im *= 255;
}
