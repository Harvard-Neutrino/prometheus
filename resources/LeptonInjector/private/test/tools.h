#include <cmath>

#include <boost/make_shared.hpp>

#include <LeptonInjector/LeptonInjector.h>

extern I3Context context;
extern const std::string earthModelName;
extern const std::string defaultCrosssectionPath;
extern const std::string defaultTotalCrosssectionPath;

extern boost::shared_ptr<I3RandomService> randomService;
extern boost::shared_ptr<earthmodel::EarthModelService> earthmodelService;

void ConfigureStandardParams(I3Module& mod);
void ConfigureEnergyRange(I3Module& mod, double minEnergy, double maxEnergy);
void resetRandomState();

//Welford's on-line variance algorithm, extended to higher moments, following
//implmentation by John D. Cook: http://www.johndcook.com/blog/skewness_kurtosis/
class MomentAccumulator{
public:
	MomentAccumulator():
	n(0),m1(0),m2(0),m3(0),m4(0){}
	
	void Insert(double x){
		const double oldN=n++;
		
		const double d = x-m1;
		const double don = d/n;
		const double don2=don*don;
		const double t = d*don*oldN;
		m1+=don;
		m4+=t*don2*(double(n)*n - 3.*n + 3) + 6*don2*m2 - 4*don*m3;
		m3+=(t*(n-2) - 3*m2)*don;
		m2+=t;
	}
	
	unsigned long NumDataValues() const{
		return n;
	}
	
	double Mean() const{
		return(m1);
	}
	
	double Variance() const{
		return((n>1) ? m2/(n-1) : 0.0);
	}
	
	double StandardDeviation() const{
		return(sqrt(Variance()));
	}
	
	double Skewness() const{
		return(sqrt(double(n)) * m3/ pow(m2, 1.5));
	}
    double Kurtosis() const{
		return n*m4 / (m2*m2) - 3.0;
	}
	
private:
	unsigned long n;
	double m1, m2, m3, m4;
};

double predictPowerLawMoment(double index, double a, double b, unsigned int moment);

void testPowerLawness(double powerlawIndex, double min, double max, unsigned int count,
					  const MomentAccumulator& moments, const std::string& file, unsigned line);

boost::shared_ptr<LeptonInjector::OutputCollector> connectCollector(I3Module& mod);