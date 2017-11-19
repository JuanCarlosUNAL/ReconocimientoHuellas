#pragma once

#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

class Filtros {
private :
	static void thinning(cv::Mat&);
	static void thinningIteration(cv::Mat&, int);
	static bool compare1(std::pair<int, double>, std::pair<int, double>);
public:
	static void filtroCompleto(cv::Mat&, cv::Mat&);
	static void caracteristicas(cv::Mat&, cv::Mat&);
	static void contorno(cv::Mat&, cv::Mat&);
	static void rellenar(cv::Mat&, cv::Mat&);
	static void recorte(cv::Mat&, cv::Mat&);

};

bool Filtros::compare1(std::pair<int, double> a, std::pair<int, double> b) {
	return a.second > b.second;
}

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

void Filtros::caracteristicas(cv::Mat& in, cv::Mat& out) {
	cv::Mat aux;
	Filtros::filtroCompleto(in, aux);

	// Caracteristicas con harris-corner detector
	cv::Mat harris_corners, harris_normalised;
	harris_corners = cv::Mat::zeros(aux.size(), CV_32FC1);
	cv::cornerHarris(aux, harris_corners, 2, 3, 0.04);
	cv::normalize(harris_corners, harris_normalised, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	float threshold = 150.0;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat rescaled;
	cv::convertScaleAbs(harris_normalised, rescaled);

	cv::Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
	cv::Mat input[] = { rescaled, rescaled, rescaled };
	int from_to[] = { 0,0, 1,1, 2,2 };

	//Guardar los puntos que cumplan con el criterio en un vector
	cv::mixChannels(input, 3, &harris_c, 1, from_to, 3);
	for (int x = 0; x<harris_normalised.cols; x++) {
		for (int y = 0; y<harris_normalised.rows; y++) {
			if ((int)harris_normalised.at<float>(y, x) > threshold) {
				// Draw or store the keypoint location here, just like you decide. In our case we will store the location of the keypoint
				cv::circle(harris_c, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), 1);
				cv::circle(harris_c, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 1);
				keypoints.push_back(cv::KeyPoint(x, y, 1));
			}
		}
	}

	out = harris_c;
}

void Filtros::rellenar(cv::Mat& img, cv::Mat& ans) {

	short kernel_morph_data[] = { 1, 1, 1 , 1, 1, 1, 1, 1, 1 };
	cv::Mat  kernel_morph(3, 3, CV_8UC1, kernel_morph_data);

	cv::Mat umbralizacion, cerradura, blur;

	cv::threshold(img, umbralizacion, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::morphologyEx(umbralizacion, cerradura, cv::MORPH_OPEN, kernel_morph, cv::Point(-1,-1), 4);

	cv::blur(cerradura, blur, cv::Size(7,7) );

	cv::bitwise_not(blur, ans);
}

void Filtros::contorno(cv::Mat& in, cv::Mat& out) {

	cv::Mat binarizada, drawing;
	std::vector<std::vector<cv::Point>> contorno;
	std::vector<cv::Vec4i> hierarchy;

	Filtros::rellenar(in, binarizada);

	cv::findContours(binarizada, contorno, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0) );

	out = cv::Mat::zeros(binarizada.size(), CV_8UC3);

	cv::RNG rng(12345);


	std::vector<std::pair<int, double>> orden;
	//buscar el contorno mas largo
	for (int i = 0; i< contorno.size(); i++) {
		double m = cv::contourArea(contorno[i]);
		orden.push_back(std::pair<int, double>(i, m));
	}

	std::sort(orden.begin(), orden.end(), Filtros::compare1);
	
	cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(out, contorno, orden[0].first, color, 2, 8, hierarchy, 0, cv::Point());

	return;
}

void Filtros::recorte(cv::Mat& in, cv::Mat& out) {
	cv::Mat binarizada, drawing;
	std::vector<std::vector<cv::Point>> contorno;
	std::vector<cv::Vec4i> hierarchy;

	Filtros::rellenar(in, binarizada);

	cv::findContours(binarizada, contorno, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	out = cv::Mat::zeros(binarizada.size(), CV_8UC3);

	cv::RNG rng(12345);


	std::vector<std::pair<int, double>> orden;
	//buscar el contorno mas largo
	for (int i = 0; i< contorno.size(); i++) {
		double m = cv::contourArea(contorno[i]);
		orden.push_back(std::pair<int, double>(i, m));
	}

	std::sort(orden.begin(), orden.end(), Filtros::compare1);

	cv::RotatedRect rr = cv::minAreaRect(contorno[orden[0].first]);
	cv::OutputArray out_array;
	cv::boxPoints(rr, out_array);
	

}