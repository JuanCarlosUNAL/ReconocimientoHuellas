#pragma once

#include "AbstractLearner.h"


class KNN : public Learner {

	cv::Ptr < cv::ml::KNearest > knn;

public:
	KNN() {
		this->knn = cv::ml::KNearest::create();
	}
};
