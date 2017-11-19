#pragma once

#include <opencv2/core/core.hpp>
#include <exception>
#include <memory>
#include <omp.h>

class Samples {
private:
	cv::Mat img;	//Imagen 
	int x_chunk;	//medida en y
	int y_chunk;	//medida en y
	int subjects;	//cantidad de clases 
	int samples;	//muestras totales en la imagen
	int n_entrenamiento; // cantidad de muestras para el entrenamiento
	void(*funcion)(cv::Mat &, cv::Mat &); //Funcion de filtro

public:
	Samples(cv::Mat &, int, int);
	cv::Mat getSample(int, int);
	cv::Mat getEntrenamiento(int);
	cv::Mat getTest();
	void setFiltro(void(*funcion)(cv::Mat &, cv::Mat &));
};

class OverloadSamples : public std::exception {
	virtual const char* what() const throw() {
		return "No existe tal muestra o sujeto.";
	}
} exception_overloadSamples;

class NoTestingSamples : public std::exception {
	virtual const char* what() const throw() {
		return "Error en las muestras de entrenamiento.";
	}
} exception_noTestingSamples;

Samples::Samples(cv::Mat &img, int subjects, int samples) {
	this->img = img;

	this->subjects = subjects;
	this->samples = samples;

	this->x_chunk = img.rows / samples;
	this->y_chunk = img.cols / subjects;

	this->funcion = NULL;
}

cv::Mat Samples::getSample(const int subject, const int sample) {
	if (subject >= this->subjects || sample >= this->samples) {
		throw exception_overloadSamples;
	}
	else {
		int xa = subject*this->y_chunk,
			ya = sample*this->x_chunk;
		if( this->funcion == NULL)
			return this->img(cv::Rect(xa, ya, this->y_chunk, this->x_chunk));
		else {
			cv::Mat ans;
			(*(this->funcion))(
				this->img(cv::Rect(xa, ya, this->y_chunk, this->x_chunk)),
				ans
				);
			return ans;
		}
	}
}

cv::Mat Samples::getEntrenamiento(int muestras = 0)  {
	if (muestras >= this->samples)
		throw exception_noTestingSamples;
	else {
		this->n_entrenamiento = (muestras > 0) ? muestras : (this->samples - 1);

		cv::Mat ans( this->n_entrenamiento * this->subjects, this->x_chunk * this->y_chunk, CV_8SC1, cv::Scalar(0));
		uchar *pt = ans.data;

		#pragma omp parallel for
		for (int i = 0; i < this->subjects; i++) {
			for (int j = 0; j < this->n_entrenamiento; j++) {
				cv::Mat aux = this->getSample(i, j);
				int size = aux.cols * aux.rows;
				std::memcpy( pt , aux.data, size );
				pt += size;
			}
		}

		cv::Mat ans_f;
		ans.convertTo(ans_f, CV_32F);

		return ans_f;
	}

}

cv::Mat Samples::getTest() {
	if (this->n_entrenamiento < 1) {
		throw exception_noTestingSamples;
	} else {
		cv::Mat ans( (this->samples - this->n_entrenamiento) * this->subjects, this->x_chunk * this->y_chunk, CV_8SC1, cv::Scalar(0));
		uchar *pt = ans.data;

		#pragma omp parallel for
		for (int i = 0; i < this->subjects; i++) {
			for (int j = this->n_entrenamiento; j < this->samples; j++) {
				cv::Mat aux = this->getSample(i, j);
				int size = aux.cols * aux.rows;
				std::memcpy(pt, aux.data, size);
				pt += size;
			}
		}
		cv::Mat ans_f;
		ans.convertTo(ans_f, CV_32F);

		return ans_f;
	}
}

void Samples::setFiltro(void(*funcion)(cv::Mat &, cv::Mat &)) {
	this->funcion = funcion;
}