/* vim: set ts=4: */
/**
 * copyright  (C) 2004
 * the icecube collaboration
 * $Id: EarthModelCalculator.h $ 
 *
 * @file EarthModelCalculator.h
 * @version $Revision: 1.16 $
 * @date $Date: 2012-03-13 10:40:52 -0500 (Tue, 13 Mar 2012) $
 * @author Kotoyo Hoshina <hoshina@icecube.wisc.edu>
 */
#ifndef EarthModelCalculator_h
#define EarthModelCalculator_h

#include <cmath>
#include <LeptonInjector/Coordinates.h>
#include <LeptonInjector/Constants.h>
#include <vector>
#include <cassert>

/**
 * @brief This is a namespace which provides a collection of stand-alone 
 * functions that calculate various geometrical information 
 * for particle propagation of the Earth. 
 */

namespace earthmodel {

namespace EarthModelCalculator
{
   enum LeptonRangeOption {DEFAULT, LEGACY, NUSIM};

  /**
   * calculate impact parameter with rewpect to origin 
   * and scalar value t that fulfills p = p0 + t*d, 
   * where p0 is start position of a track, d is 
   * direction of the track, and p is most closest position
   * on a track from origin.
   * this function should work in any Cartecian coordinate
   * 
   * @param[in] p0  particle start position
   *
   * @param[in] d  particle direction (unit vector)
   *
   * @param[out] t distance from p0 to p
   *
   * @param[out] p most closest position on a track from origin
   *
   * @return impact parameter, distance from origin to p
   *
   */
   double GetImpactParameter(const LeptonInjector::LI_Position &p0,
                             const LeptonInjector::LI_Direction &d,
                             double &t,
                             LeptonInjector::LI_Position &p);

  /**
   * This function returns intersection-positions between a track
   * and a sphere with radius r.
   * Note that the origin of track position and direction must be 
   * at the center of the sphere. Return positions are in same
   * coordinate as input.
   * If there is only one intersection, it will be stored in both
   * output parameters.
   * 
   * @param[in] pos track position 
   * @param[in] dir track direction (unit vector)
   * @param[in] r   radius
   *
   * @param[out] startPos  track-entering-position to the sphere
   * @param[out] endPos    track-exitting-position from the sphere
   *
   * @return number of intersections
   */
   int GetIntersectionsWithSphere(
                           const LeptonInjector::LI_Position &pos,
                           const LeptonInjector::LI_Direction &dir,
                           double r,
                           LeptonInjector::LI_Position &startPos,
                           LeptonInjector::LI_Position &endPos);

  /**
   * wrapper function of GetIntersectionsWithSphere
   * 
   * @param[in] pos track position 
   * @param[in] dir track direction (unit vector)
   * @param[in] r   radius
   *
   * @param[out] enterdist distance from pos to track-entering-position
   *             negative value indicates behind the pos
   *
   * @param[out] exitdist distance from pos to track-exiting-position
   *             negative value indicates behind the pos
   *
   * @return number of intersections
   */
   int GetDistsToIntersectionsWithSphere(
                           const LeptonInjector::LI_Position &pos,
                           const LeptonInjector::LI_Direction &dir,
                           double r,
                           double & enterdist,
                           double & exitdist);

  /**
   * @brief Returns muon range in m.w.e.
   * If you need surviving length [m] of muon/tau, use
   * EarthModelService::GetLeptonRangeInMeter(). 
   *
   * @return range [m.w.e]
   *
   * Now MuonRange calculation offers three options:
   *
   * DEFAULT
   *    -> use Dima's fitting function and optimized parameter
   *       confirmed on Mar. 29, 2011
   *
   * LEGACY
   *    -> use legacy nugen equation and parameter
   *       obtained study with mmc, using ice medium
   *
   * NUSIM
   *    -> use Gary's fitting function (used in NUSIM)
   *
   * scale gives 
   * a scaling factor for MuonRange in Meter.
   * See GetLeptonRangeInMeter().
   *
   * [Dima's equation]
   *
   * muon and tau: R = (1/b)*log[(bE/a)+1]
   * for the equation see arXiv:hep-ph/0407075, section 6.2 P16.
   *
   * muon: (legacy nugen parameter)  
   * b~3.4*10^-4 [1/m]
   * a~2*10^-1 [GeV/m]
   * R_mu ~ 3000 * log(1.0 + 1.7*10^-3*E[GeV]) [mwe]
   *
   * tau:  (see: http://www.ice.phys.psu.edu/~toale/icecube/nutau/nutau-mc.html)
   * b~2.6*10^-5 [1/m]
   * a~1.5*10^3 [GeV/m]
   * R_tau ~ 38000 * log(1.0 + 1.8*10^-8*E[GeV]) [mwe]
   *
   *
   * [Gary's equation] (muon only)
   *
   * if (energy < 1e3) {
   *    R_mu = energy/0.002;
   * } else if (energy < 1e4) {
   *    R_mu = 1.0e5 * (3.5 + 9.0 * (TMath::Log10(energy) - 3.0));
   * } else {
   *    R_mu = 1.0e5 * (12.0 + 6.0 * (TMath::Log10(energy) - 4.0));
   * }
   *
   */
   double GetLeptonRange(double particle_energy, 
                       bool   isTau = false,
                       LeptonRangeOption option = DEFAULT,
                       double scale = 1);
    
   /**
    * @brief unit conversion from g/cm2 to m.w.e
    */
   double ColumnDepthCGStoMWE(double cdep_CGS);

   /**
    * @brief unit conversion from m.w.e to g/cm2
    */
   double MWEtoColumnDepthCGS(double range_MWE);

}
   
namespace Integration{

   /**
    * An object which performs 1-D intgration of a function
    * using the trapezoid rule, and allows the approximation
    * to be refined incrementally.
    */
   template<typename FuncType>
   struct TrapezoidalIntegrator{
   private:
      /**
       * The function being integrated
       */
      const FuncType& f;
      /**
       * The lower bound of integration
       */
      double a;
      /**
       * The upper bound of integration
       */
      double b;
      /**
       * Counter expressing the number of times the integral
       * approximation has been refined
       */
      unsigned int currentDetail;
      /**
       * The current approximation of the integral
       */
      double value;
      
      /**
       * Add one level of detail to the integral approximation
       */
      void update(){
         currentDetail++;
         if(currentDetail==1)
			value=(b-a)*(f(a)+f(b))/2;
         else{
			unsigned long npoints=1ul<<(currentDetail-2);
			double dx=(b-a)/npoints;
			double x=a+dx/2;
			double sum=0.0;
			for(unsigned long i=0; i<npoints; i++, x+=dx)
               sum+=f(x);
			value=(value+(b-a)*sum/npoints)/2;
         }
      }
      
   public:
      /**
       * @param f the function to be integrated
       * @param a the lower bound of integration
       * @param b the upper bound of integration
       */
      TrapezoidalIntegrator(const FuncType& f, double a, double b):
      f(f),a(a),b(b),currentDetail(0),value(0){
         if(a>b)
			std::swap(a,b);
      }
      
      /**
       * Get the integral approximation, updating it with higher 
       * detail if necessary
       *
       * @param detail how finely to approximate the integral.
       *               A detail level of n requires 1+2^n function evaluations, 
       *               but reuses any evaluations already performed when lower
       *               detail levels were calculated.
       */
      double integrate(unsigned int detail){
         detail++;
         while(currentDetail<detail)
			update();
         return(value);
      }
      
      /**
       * Get the current detail level of the ingral approximation.
       */
      unsigned int getDetail() const{ return(currentDetail); }
   };

   
   /**
    * @brief Performs a fifth order Romberg integration of a function to a chosen tolerance. 
    *
    * This routine is rather simplistic and not suitable for complicated functions, 
    * particularly not ones with discontinuities, but it is very fast for smooth functions.
    *
    * @param func the function to integrate
    * @param a the lower bound of integration
    * @param b the upper bound of integration
    * @param tol the (absolute) tolerance on the error of the integral
    */
   template<typename FuncType>
   double rombergIntegrate(const FuncType& func, double a, double b, double tol=1e-6){
      const unsigned int order=5;
      const unsigned int maxIter=20;
      if(tol<0)
         throw("Integration tolerance must be positive");
      
      std::vector<double> stepSizes, estimates, c(order), d(order);
      stepSizes.push_back(1);
      
      TrapezoidalIntegrator<FuncType> t(func,a,b);
      for(unsigned int i=0; i<maxIter; i++){
         //refine the integral estimate
         estimates.push_back(t.integrate(t.getDetail()));
         
         if(i>=(order-1)){ //if enough estimates have been accumulated
			//extrapolate to zero step size
			const unsigned int baseIdx=i-(order-1);
			std::copy(estimates.begin()+baseIdx,estimates.begin()+baseIdx+order, c.begin());
			std::copy(estimates.begin()+baseIdx,estimates.begin()+baseIdx+order, d.begin());
			
			double ext=estimates.back(), extErr;
			for(unsigned int m=1; m<order; m++){
               for(unsigned int j=0; j<order-m; j++){
                  double ho=stepSizes[j+baseIdx];
                  double hp=stepSizes[j+m+baseIdx];
                  double w=c[j+1]-d[j];
                  double den=ho-hp;
                  assert(den!=0.0);
                  den=w/den;
                  c[j]=ho*den;
                  d[j]=hp*den;
               }
               extErr=d[order-1-m];
               ext+=extErr;
			}
			
            //declare victory if the tolerance criterion is met
			if(std::abs(extErr)<=tol*std::abs(ext))
               return(ext);
         }
         //prepare for next step
         stepSizes.push_back(stepSizes.back()/4);
      }
      throw("Integral failed to converge");
   }
   
}

}

#endif
