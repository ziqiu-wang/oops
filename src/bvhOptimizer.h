#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace oops {

	// k_opt = optimal size step for rebuilding bvh
	// tr = cost of rebuild (ms)
	// tu = cost of refit (ms)
	// dq = per time step growth of query time of refitting
	
	class bvhOptimizer {
	public: 
		int initialInterval = 8; // start at this while we have no information 
		int minInterval = 2;
		int maxInterval = 500;
		int minSamples = 4; // at least this many samples to measure dq (change in bvh quality)
		float smoothing = 0.3; // weight for ema

		bool rebuild() const{
			return lastRebuilt >= interval();
		}

		void update(bool rebuilt, float accelMs, float queryMs) {
			if (rebuilt) {
				if (rebuiltBefore) {
					tr = expMovingAvg(tr, accelMs);
				}
				else {
					tr = accelMs;
				}
				rebuiltBefore = true;
				lastRebuilt = 0;
				queryTimes.clear();
			}
			else {
				if (refitBefore) {
					tu = expMovingAvg(tu, queryMs);
				}
				refitBefore = true;
				lastRebuilt++; // number of time steps 
			}
			queryTimes.push_back(queryMs);
			if (queryTimes.size() >= minSamples) {
				dq = slope(); // calculate slope
			}
		}

		int interval() const {
			if (!rebuiltBefore ||
				!refitBefore ||
				(!dqMeasured || queryTimes.size() < minSamples)) {
				return initialInterval;
			}
			else {
				if (tr <= tu) return minInterval; // rebuild more bc cheaper than refitting
				if (dq <= 1e-9) return maxInterval; // refit more bc less degredation
				float kopt = -1.0 + std::sqrt(1.0 + 2.0 * (tr - tu) / dq);
				return std::clamp(static_cast<int>(std::round(kopt)), minInterval, maxInterval);
			}
		}

		// for GUI display
		float rebuildTime() const {
			return tr;
		}

		float refitTime() const {
			return tu;
		}

		float getDq() const {
			return dq;
		}

		int refitsSinceRebuild() const {
			return lastRebuilt;
		}

	private:
		
		float expMovingAvg(float old, float sample) const {
			return (1.0 - smoothing) * old + smoothing * sample;
		}

		float slope() {
			dqMeasured = true;
			float n = float(queryTimes.size());
			float sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
			for (int j = 0; j < queryTimes.size(); ++j) {
				float y = queryTimes[j];
				float x = float(j);
				sumX += x;
				sumY += y;
				sumXY += x * y;
				sumXX += x * x;
			}
			double denom = n * sumXX - sumX * sumX;
			if (denom > 0) {
				return (n * sumXY - sumX * sumY) / denom;
			}
			return 0.0;
		}

		float tr = 0, tu = 0, dq = 0;
		bool rebuiltBefore = false, refitBefore = false, dqMeasured = false;
		int lastRebuilt = 0;
		std::vector<float> queryTimes;
	};
}