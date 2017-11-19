#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>

#include "samples.h"
#include "Filtros.h"

#include <iostream>

using namespace std;

int main2(int argc, char** argv) {
	
	//Configuraciones
	omp_set_num_threads(4);

	//Verificar los argumentos
	if (argc != 2){
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}
	//Leer la imagen 
	cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

	if (!input.data) { // Check for invalid input
		cout << "no se pudo abrir o encontrar la imagen." << std::endl;
		return -2;
	}


	//Obtener y dividir la imagen
	cout << "Obtener conjuntos de prueba." << endl;
	Samples samples(input, 7, 8);
	
	//Introducir el filtro
	samples.setFiltro(Filtros::filtroCompleto);
	
	//Datos de entrenamiento
	cv::Mat train_samples = samples.getEntrenamiento();
	cv::Mat train_labels(train_samples.rows, 1, CV_32FC1);
	for (int i = 0; i < train_samples.rows; i++)
		train_labels.row(i) = cv::Scalar(i / 7);

	// Datos de test
	cv::Mat test_samples = samples.getTest();
	cv::Mat test_labels(train_samples.rows, 1, CV_32FC1);
	for (int i = 0; i < test_samples.rows; i++)
		test_labels.row(i) = cv::Scalar(i / 7);

	// Algoritmo de entrenamiento
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

	knn->train(train_samples, cv::ml::ROW_SAMPLE, train_labels);

	cv::Mat responses;
	knn->findNearest(test_samples, 7, responses);

	cout << responses << endl;

	return 0;
}

int main(int argc, char** argv) {
	//Configuraciones
	omp_set_num_threads(4);

	//Verificar los argumentos
	if (argc != 2) {
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}
	//Leer la imagen 
	cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

	if (!input.data) { // Check for invalid input
		cout << "no se pudo abrir o encontrar la imagen." << std::endl;
		return -2;
	}


	//Obtener y dividir la imagen
	cout << "Obtener conjuntos de prueba." << endl;
	Samples samples(input, 7, 8);

	//Introducir el filtro
	samples.setFiltro(Filtros::recorte);

	cv::imshow("imagen", samples.getSample(1, 6));

	cv::waitKey(0); // Wait for a keystroke in the window

	return 0;
}

int main3(int argc, char** argv) {

	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

	if (!input.data) // Check for invalid input
	{
		cout << "no se pudo abrir o encontrar la imagen." << std::endl;
		return -2;
	}

	Samples samples(input, 7, 8);

	samples.setFiltro(Filtros::filtroCompleto);

	cv::imshow("ejemplo", samples.getSample(5,0));

	// Puntos de interes
	cv::Mat harris_corners, harris_normalised;
	harris_corners = cv::Mat::zeros(samples.getSample(5,0).size(), CV_32FC1);
	cv::cornerHarris(samples.getSample(5, 0), harris_corners, 2, 3, 0.04);
	cv::normalize(harris_corners, harris_normalised, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	float threshold = 150.0;
	vector<cv::KeyPoint> keypoints;
	cv::Mat rescaled;
	cv::convertScaleAbs(harris_normalised, rescaled);

	cv::Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
	cv::Mat in[] = { rescaled, rescaled, rescaled };
	int from_to[] = { 0,0, 1,1, 2,2 };

	cv::mixChannels(in, 3, &harris_c, 1, from_to, 3);
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

	cv::Ptr<cv::Feature2D> orb_descriptor = cv::ORB::create();
	cv::Mat descriptors;
	orb_descriptor->compute(samples.getSample(5, 0), keypoints, descriptors);

	// We will still need to fill those once we compare everything, by using the code snippets above
	vector<cv::Mat> database_descriptors;
	cv::Mat current_descriptors;
	// Create the matcher interface
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	// Now loop over the database and start the matching
	vector< vector< cv::DMatch > > all_matches;
	for (int entry = 0; entry < database_descriptors.size(); entry++) {
		vector< cv::DMatch > matches;
		matcher->match(database_descriptors[entry], current_descriptors, matches);
		all_matches.push_back(matches);
	}

	cv::imshow("Puntos de Interes", harris_c); // Show our image inside it.
	cv::imwrite("puntos_interes.jpg", harris_c);

	//cv::namedWindow("Filtros", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Filtros", samples.getSample(5, 0)); // Show our image inside it.

	cv::waitKey(0); // Wait for a keystroke in the window

	return 0;

	return 0;
}

int huy(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

	if (!input.data) // Check for invalid input
	{
		cout << "no se pudo abrir o encontrar la imagen." << std::endl;
		return -2;
	}

	cv::Mat img_filtrada;
	Filtros::filtroCompleto(input, img_filtrada);
	
	// Puntos de interes
	cv::Mat harris_corners, harris_normalised;
	harris_corners = cv::Mat::zeros(img_filtrada.size(), CV_32FC1);
	cv::cornerHarris(img_filtrada, harris_corners, 2, 3, 0.04);
	cv::normalize(harris_corners, harris_normalised, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );

	float threshold = 150.0;
	vector<cv::KeyPoint> keypoints;
	cv::Mat rescaled;
	cv::convertScaleAbs(harris_normalised, rescaled);
	
	cv::Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
	cv::Mat in[] = { rescaled, rescaled, rescaled };
	int from_to[] = { 0,0, 1,1, 2,2 };
	
	cv::mixChannels(in, 3, &harris_c, 1, from_to, 3);
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

	cv::Ptr<cv::Feature2D> orb_descriptor = cv::ORB::create();
	cv::Mat descriptors;
	orb_descriptor->compute(img_filtrada, keypoints, descriptors);

	// We will still need to fill those once we compare everything, by using the code snippets above
	vector<cv::Mat> database_descriptors;
	cv::Mat current_descriptors;
	// Create the matcher interface
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	// Now loop over the database and start the matching
	vector< vector< cv::DMatch > > all_matches;
	for (int entry = 0; entry < database_descriptors.size(); entry++) {
		vector< cv::DMatch > matches;
		matcher->match(database_descriptors[entry], current_descriptors, matches);
		all_matches.push_back(matches);
	}

	cv::imshow("Puntos de Interes", harris_c); // Show our image inside it.
	cv::imwrite("puntos_interes.jpg", harris_c);

	//cv::namedWindow("Filtros", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Filtros", img_filtrada); // Show our image inside it.

	cv::waitKey(0); // Wait for a keystroke in the window
	
	return 0;
}
