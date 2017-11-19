#pragma once

#include <opencv2/core/core.hpp>
#include <exception>
#include <opencv2\core.hpp>
#include <opencv2\ml.hpp>

class Learner {
public:
	Learner() {}
	virtual int entrenar(cv::Mat, cv::Mat);
	virtual int clasificar(cv::Mat);
};

class AbstractClass : public std::exception {
	virtual const char* what() const throw() {
		return "Metodos no implementados.";
	}
} exception_abstractClass;

int Learner::entrenar(cv::Mat muestras, cv::Mat etiquetas) {
	throw exception_abstractClass;
	return -1;
}

int Learner::clasificar(cv::Mat muestras) {
	throw exception_abstractClass;
	return -1;
}