#pragma once

#include "AbstractLearner.h"
#include "KNN.h"
#include "K_means.h"

class LearnerFactory {
public:
	static Learner getKNN() {
		return KNN();
	}
	static Learner getKMeans() {
		return KMeans();
	}
};