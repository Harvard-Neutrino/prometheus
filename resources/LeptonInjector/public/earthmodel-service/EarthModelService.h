#ifndef LI_EarthModelService_H  
#define LI_EarthModelService_H  
/**
 *@file EarthModelService.h
 *@brief A class for managing Earth's density profile and ice geometry.
 *
 * It also offers utility functions that may be useful for propagators.
 *
 *  - GetDensityInCGS(pos) function
 *  - GetMedium(pos) function (currently returns ROCK or ICE)
 *  - Convert position & direction in Earth-centered coordinate to 
 *    Detector Coordinate, and vice vasa
 *  
 * This class requires a crust data file (ascii format).
 * see resources/earthparam/PREM_mmc.dat for details
 * 
 * CAUTION :
 * The internal units of the module are [m] for length and
 * [g/cm^3] for density, as in the data format mentioned above.
 * The unit conversion is done inside the module so that 
 * you need to set density etc. with IceCube units.
 *
 *@author Kotoyo Hoshina (hoshina@icecube.wisc.edu)
 *@version $Id: $
 *@date $Date: $
 *(c) the IceCube Collaboration
 *
 */

// include custom LI headers
#include <LeptonInjector/Constants.h>
#include <LeptonInjector/Coordinates.h>

// some standard lib headers
#include <sstream>
#include <string>
#include <assert.h>
#include <vector>
#include <map>
#include <iostream>

// earthmodel! 
#include <earthmodel-service/EarthModelCalculator.h>

namespace earthmodel {

//______________________________________________________________
class EarthModelService
{

 public:

   enum MediumType {INNERCORE=1, 
                    OUTERCORE=2, 
                    MANTLE=3, 
                    ROCK=4, 
                    ICE=5, 
                    AIR=6, 
                    VACUUM=7, 
                    WATER=8,
                    LOWERMANTLE = 31,
                    UPPERMANTLE = 32,
                    LLSVP = 33  // Large Low Shear Velocity Provinces
                    };

   enum IceCapType {NOICE, ICESHEET, SIMPLEICECAP};

   enum IntegType {PATH, RADIUS, CIRCLE, SPHERE};

   typedef std::map<MediumType, std::map<int, double> > MatRatioMap;
   
   /**
    * A representation of one spherical shell of material.
    */
   class EarthParam
   {
   public:
      EarthParam() {}
      EarthParam(const EarthParam &p) : 
         fUpperRadius_(p.fUpperRadius_), fZOffset_(p.fZOffset_),
         fBoundaryName_(p.fBoundaryName_), fMediumType_(p.fMediumType_)
         { std::copy(p.fParams_.begin(), p.fParams_.end(), back_inserter(fParams_) ); }

      /**
       * The outer radius of this shell.
       *
       * Note that the inner radius is not stored here; it is either zero
       * or the outer radius of the next shell inwards.
       */
      double              fUpperRadius_;
      /**
       * The offset (towards to south pole) of the center of this shell
       * from the center of the earth.
       */
      double              fZOffset_;
      /**
       * The name of the upper boundary of this shell
       */
      std::string         fBoundaryName_;
      /**
       * The type of material from which this shell is composed
       */
      MediumType          fMediumType_;
      /**
       * The coefficients of the polynomial which gives the density as a
       * function of radius in this shell
       */
      std::vector<double> fParams_;
      
      /**
       * Evaluates the density within this shell
       *
       * @param x the radial coordinate in meters
       * @return the material density in g/cm^3
       */
      const double GetDensity(double x) const
      {
         unsigned int n = fParams_.size() -1;
         double den = fParams_[n];
         while (n>0) den = den * x + fParams_[--n];
         return den;
      }  

      const double GetDensityGrad(double x) const
      {
         unsigned int n = fParams_.size() -1;
         double den = fParams_[n];
         double res = 0;
         while (n>1) {
             den = den * x + fParams_[--n];
             res += den;
         }
         return res;
      }  

      /**
       * print density in density file format
       */
      std::string PrintDensity() const
      {
         std::stringstream ss;
         ss << fUpperRadius_ << "\t" << fBoundaryName_ << "\t" 
            << EarthModelService::GetMediumTypeString(fMediumType_)
            << "\t" << fParams_.size() << "\t";
         for (unsigned int n=0; n<fParams_.size(); ++ n) {
            ss << fParams_[n] << "\t";
         }
         return ss.str();
      }

   };

   typedef  std::map<double, EarthParam> EarthParamMap;

   /** 
    *@brief Constructor - 
    */
   EarthModelService();

   /** 
    *@brief constructor for pybinding
    * @param [in] name : (dummy) name to pass to I3ServiceBase
    * @param [in] tablepath : path to table dir. blank: use default path
    * @param [in] earthmodels : list of earthmodel files. 
    *             You may add muliple files, and latter one overwrites
    *             former value. e.g. PREM_mmc + FLATCORE = PREM earth
    *             with flat density at core
    * @param [in] materialmodels : list of material files. 
    *             You may add muliple files, and latter one overwrites
    *             former value.
    * @param [in] icecapname : type of icecap shape.
    * @param [in] icecapangle: valid only when SimpleIceCap is selected
    * @param [in] detectordepth : depth of icecube center
    *
    */
   EarthModelService(
          const std::string& name,
          const std::string &tablepath = "",
          const std::vector<std::string>& earthmodels = std::vector<std::string>(),
          const std::vector<std::string>& materialmodels = std::vector<std::string>(),
          const std::string &icecapname = "SimpleIceCap",
          double icecapangle = 20.0*LeptonInjector::Constants::degrees,
          double detectordepth = 1948.0*LeptonInjector::Constants::m);

   /**
    * Destructor - 
    */
   virtual ~EarthModelService();

   /**
    * configure - unnecessary in standalone
    */  
   //void Configure();

   //-----------------
   // main interfaces
   //-----------------

   static std::string GetMediumTypeString(MediumType m);
   static MediumType  ConvertMediumTypeString(const std::string &s);
   
   /**
    * @brief Finds the material layer containing p_CE
    *
    * @param p_CE the position for which to find the material layer, which
    *             must be specified in Earth-centered coordinates.
    */
   const EarthParam& GetEarthParam(const LeptonInjector::LI_Position& p_CE) const;


   /**
    * @brief Computes the material density in the given layer. 
    * 
    * This function is useful when multiple density queries are needed which 
    * are known to all be within the same layer, since the layer can be cached 
    * and reused for each query. It is the user's responsibility to ensure that 
    * ep is actually the correct material layer for p_CE (most likey by ensuring 
    * that it is the result of a call to `GetEarthParam(p_CE)`).
    *
    * @param ep the material layer in which the density should be computed
    * @param posi3 the position at which the density is to be evaluated. 
    *             This must be in detector-centered coordinates.
    * @return the density in g/cm^3
    */
   const double GetDensityInCGS(const EarthParam& , const LeptonInjector::LI_Position &);

   /**
    * @brief Computes the material density at the given point.
    *
    * @param posi3 the position at which the density is to be evaluated.
    *             This must be in detector-centered coordinates.
    * @return the density in g/cm^3
    */
   const double GetDensityInCGS(const LeptonInjector::LI_Position &posi3) const;

   /**
    * @brief Computes the material density at the given point.
    *
    * @param p_CE the position at which the density is to be evaluated.
    *             This must be in Earth-centered coordinates.
    * @return the density in g/cm^3
    */
   const double GetEarthDensityInCGS(const LeptonInjector::LI_Position& p_CE) const;
   
   /**
    * @brief Computes the material density in the given layer. 
    * 
    * This function is useful when multiple density queries are needed which 
    * are known to all be within the same layer, since the layer can be cached 
    * and reused for each query. It is the user's responsibility to ensure that 
    * ep is actually the correct material layer for p_CE (most likey by ensuring 
    * that it is the result of a call to `GetEarthParam(p_CE)`).
    *
    * @param ep the material layer in which the density should be computed
    * @param p_CE the position at which the density is to be evaluated. 
    *             This must be in Earth-centered coordinates.
    * @return the density in g/cm^3
    */
   static double GetEarthDensityInCGS(const EarthParam& ep, const LeptonInjector::LI_Position& p_CE);
   
   /**
    * GetColumnDepthInCGS
    *
    * @brief This function calculates total column depth
    * from from_posI3 to to_posI3.
    *
    * @param[in] from_posI3 from position in I3 coordinate
    *
    * @param[in] to_posI3 to position in I3 coordinate
    */
   const double GetColumnDepthInCGS(
                    const  LeptonInjector::LI_Position &from_posI3,
                    const  LeptonInjector::LI_Position &to_posI3,
                    const bool use_electron_density=false) const;

   /**
    * IntegrateDensityInCGS
    *
    * @brief integrate density 
    *
    * @param[in] from_posCE from position in Earth centered coordinate
    *
    * @param[in] to_posCE to position in Earth centered coordinate
    *
    * @param[in] intg_type : integrate density with
    *  PATH : along a path from from_posCE to to_posCE. [g/cm2] 
    *  RADIUS : similar to PATH but projected on radial direction [g/cm2]
    *  CIRCLE : 2*pi*r x RADIUS option [g/cm]
    *  SPHERE : 4*pi*r^2 x RADIUS option (volume mass)[g]
    *
    * @param[in] use_electron_density to use a scaling factor on the density, yielding an adjusted column density based off of the electron number density 
    */
   const double IntegrateDensityInCGS(
                    const  LeptonInjector::LI_Position &from_posCE,
                    const  LeptonInjector::LI_Position &to_posCE, 
                    IntegType intg_type = PATH,
                    const  bool use_electron_density = false) const;


    const std::vector<std::tuple<double,double,double>> GetDensitySegments(
                 const  LeptonInjector::LI_Position &from_posI3,
                 const  LeptonInjector::LI_Position &to_posI3) const;

    const std::vector<std::tuple<double,double,double>> GetEarthDensitySegments(
                 const  LeptonInjector::LI_Position &from_posCE,
                 const  LeptonInjector::LI_Position &to_posCE) const;
 
   
   /**
    * @brief Computes the distance along the given direction, ending at the given point,
    *        which must be traversed to accumulate the specified column depth.
    *
    * @param to_posI3 the endpoint of the path
    * @param dirI3 the direction of the path
    * @param cDepth the column depth required in g/cm^2
    * @return the distance in meters
    */
   double DistanceForColumnDepthToPoint(
                    const  LeptonInjector::LI_Position& to_posI3,
                    const  LeptonInjector::LI_Direction& dirI3,
                    double cDepth,
                    const  bool use_electron_density = false) const;
   
   /**
    * @brief Computes the distance along the given direction, starting at the given point,
    *        which must be traversed to accumulate the specified column depth.
    *
    * @param from_posI3 the starting point of the path
    * @param dirI3 the direction of the path
    * @param cDepth the column depth required in g/cm^2
    * @return the distance in meters
    */
   double DistanceForColumnDepthFromPoint(
                    const  LeptonInjector::LI_Position& from_posI3,
                    const  LeptonInjector::LI_Direction& dirI3,
                    double cDepth,
                    const bool use_electron_density=false) const;

   /**
    * GetLeptonRangeInMeterFrom
    *
    * @brief This function calculates MuonRange [m.w.e] and convert
    * it to distance [m] with given particle info, earth model and 
    * the start position (o)
    *
    *           d[m]
    *   |----------------|
    *   <----------------o
    *      dir       start pos
    *
    * @param[in] posI3 start position to calculate range
    *
    * @param[in] dirI3 direction where the particle is moving to (particle direction)
    *
    * @param[in] energy particle energy
    *
    * @param[in] isTau if set the lepton will be treated as a tau, otherwsie as a muon.
    * 
    * @param[in] opt, scale
    * Used by EarthModelCalculator::GetLeptonRange() function.
    * Leave it as defaut unless you understand well about
    * the parameters.
    *
    * @return distance d[m]
    *
    */
   const double GetLeptonRangeInMeterFrom(
                    const  LeptonInjector::LI_Position &posI3,
                    const  LeptonInjector::LI_Direction &dirI3,
                    double energy,
                    bool   isTau = false,
                    EarthModelCalculator::LeptonRangeOption 
                           opt = EarthModelCalculator::DEFAULT,
                    double scale = 1.0) const;

   /**
    * GetLeptonRangeInMeterTo
    *
    * @brief This function calculates MuonRange [m.w.e] and convert
    * it to distance [m] with given particle info, earth model and 
    * the end position (o)
    *
    *           d[m]
    *   |-----------------|
    *   o<----------------|
    * end pos     dir      
    *
    * @param[in] posI3 end position of the conversion
    *
    * @param[in] dirI3 direction where the particle is moving to (particle direction)
    *
    * @param[in] energy particle energy
    *
    * @param[in] isTau if set the lepton will be treated as a tau, otherwsie as a muon.
    * 
    * @param[in] opt, scale
    * Used by EarthModelCalculator::GetLeptonRange() function.
    * Leave it as defaut unless you understand well about
    * the parameters.
    *
    * @return distance d[m]
    *
    */
   const double GetLeptonRangeInMeterTo(
                    const  LeptonInjector::LI_Position &posI3,
                    const  LeptonInjector::LI_Direction &dirI3,
                    double energy,
                    bool   isTau = false,
                    EarthModelCalculator::LeptonRangeOption 
                           opt = EarthModelCalculator::DEFAULT,
                    double scale = 1.0) const;
   
   /**
    * @brief Computes the disance to the next boundary between material
    *        layers along the given track.
    *
    * @param posCE the starting point of the track in EarthCenter coordinate
    * @param dirCE the direction of the track in EarthCenter coordinate
    * @param[out]  exitsAtmosphere whether the next boundary crossing will
    *              be the outer surface of the atmosphere. This parameter
    *              is optional and may be omitted if this information is
    *              not of interest
    */
   double DistanceToNextBoundaryCrossing(const LeptonInjector::LI_Position& posCE,
                                         const LeptonInjector::LI_Direction& dirCE,
                                         bool& exitsAtmosphere=ignored_bool) const;


   /**
    * Get medium type at a given point.
    * YOU MUST USE Earth-centered coordinate for argument.
    */
   const MediumType GetMedium(const LeptonInjector::LI_Position &p_CE) const;

   /**
    * @brief Get material ratio map (for genie)
    */
   const MatRatioMap GetMatRatioMap() const {return fMatRatioMap_; }
   const std::map<int, double>& GetMatRatioMap(MediumType med) const;
   double GetMatRatio(MediumType med, int id) const;

   /**
    * @brief Get ratio map of proton, neutron and electron (for nugen)
    */
   const MatRatioMap GetPNERatioMap() const {return fPNERatioMap_; }
   const std::map<int, double>& GetPNERatioMap(MediumType med) const;
   double GetPNERatio(MediumType med, int id) const;

   /**
    * Get track path length from Earth entrance point of track
    * to detector center
    */
   const double GetDistanceFromEarthEntranceToDetector(double zenith_rad) const;
   const std::vector<double> GetDistanceFromSphereSurfaceToDetector(
                             double zenrad,
                             double radius,
                             double det_z) const;

   /**
    * Printing Earth Params
    */
   std::string PrintEarthParams() const;

   /**
    * Get Earth Params.
    */ 
    //currently borked
    /*
   boost::python::list GetEarthParamsList() 
   { 
      boost::python::list py_list;
      BOOST_FOREACH(EarthParamMap::value_type param, fEarthParams_) {
         py_list.append(param.second);
      }
      return py_list;
   } 
   */

   /**
    * Earth model function. 
    * CAUTION : Return unit is [g/cm3] !
    */
   const double GetPREM(double r) const; 


   //-----------------
   // util functions 
   //-----------------

   // position conversion 

   /**
    * convert LI_Position in detector coordinate to Earth Center Coordinate
    */
   const LeptonInjector::LI_Position GetEarthCoordPosFromDetCoordPos(const LeptonInjector::LI_Position &p) const;

   /**
    * convert LI_Position in Earth Center Coordinate to Detector Coordinate
    */
   const LeptonInjector::LI_Position GetDetCoordPosFromEarthCoordPos(const LeptonInjector::LI_Position &p) const;

   // direction conversion 

   /**
    * convert I3Direction in Detector Coordinate to Earth Center Coordinate
    */
   const LeptonInjector::LI_Direction GetEarthCoordDirFromDetCoordDir(const LeptonInjector::LI_Direction &p) const;

   /**
    * convert I3Direction in Earth Center Coordinate to Detector Coordinate
    */
   const LeptonInjector::LI_Direction GetDetCoordDirFromEarthCoordDir(const LeptonInjector::LI_Direction &p) const;

   //---------
   // setters
   //---------

   /**
    * Set path of parameter files
    */
   inline void SetPath(const std::string& s) { fPath_ = s; }

   /**
    * Set name of Earth Model
    */
   void SetEarthModel(const std::vector<std::string> & s); 

   /**
    * Set Material Model
    */
   void SetMaterialModel(const std::vector<std::string> & s); 

   /**
    * Set IceCap type
    *  - "NoIce" ... no ice at all
    *  - "IceSheet" ... use ice sheet wrapps entirely the Earth 
    *  - "SimpleIceCap" ... use simple spherical dorm ice
    */
   void SetIceCapTypeString(std::string s);

   /**
    * Set open angle of ice cap
    */
   void SetIceCapSimpleAngle(double cap_angle); 

   /**
    * Set detector depth 
    */
   void SetDetectorDepth(double d);

   /**
    * Set detector XY
    */
   void SetDetectorXY(double x, double y);

   //----------
   // getters
   //----------

   /**
    * Get Detector depth.
    */
   const double GetDetectorDepth() const {return fDetDepth_;}

   /**
    * Get Detector coordinate origin with respect to Earth Center Coordinate
    */
   const LeptonInjector::LI_Position & GetDetectorPosInEarthCoord() const {return fDetPos_;}

   /**
    * get path of parameter files
    */
   const std::string& GetPath() const { return fPath_ ; }

   /**
    * Get open angle of ice cap
    */
   const std::string& GetIceCapTypeString() const { return fIceCapTypeString_; }

   /**
    * Get open angle of ice cap
    */
   const double GetIceCapSimpleAngle() const { return fIceCapSimpleAngle_; }

   /**
    * Get Boundary [m]
    */
   const double GetBoundary(const std::string &s) const;


   /**
    * Get MohoBoundary [m]
    * This may be -1 if the model has no moho boundary
    */
   const double GetMohoBoundary() const { return fMohoBoundary_; }

   /**
    * Get Rock-Ice boundary [m]
    * This will be the outer radius of the uppermost layer of rock if no ice is present
    */
   const double GetRockIceBoundary() const { return fOutermostRockBoundary_; }

   /**
    * Get the outer radius of the uppermost layer of rock [m]
    * This may be 0 if the model has no rock.
    */
   double GetOutermostRockBoundary() const { return fOutermostRockBoundary_; }


   /**
    * Get Ice-Air boundary [m]
    * If you have simple cap ice, this value is most far Ice-Air boundary
    * from the center of the Earth.
    * If there is no ice, this is the outer radius of the uppermost layer
    * which is not part of the atmosphere [m]
    */
   const double GetIceAirBoundary() const { return fEarthAirBoundary_; }

   /**
    * Get the outer radius of the uppermost layer which is not part of the atmosphere [m]
    * This may be 0 if the model has no layers denser than air.
    */
   double GetEarthAirBoundary() const { return fEarthAirBoundary_; }

   /**
    * Get Radius of atmosphare [m]
    */
   const double GetAtmoRadius() const { return fAtmoRadius_; }

   /**
    * Convert radius to coszen of tangent line of the radius.
    * This function always return 1 if the radius exceeds
    * distance from center of the Earth to IceCube center
    */
   const double RadiusToCosZen(double r) const;

   /**
    * init function
    */
   void Init();

 private:

   /**
    * GetLeptonRangeInMeter
    *
    * @brief This function converts MuonRange in m.w.e to m
    * with given earth model and geometry of the track.
    *
    * @param[in] energy particle energy
    *
    * @param[in] posI3 anchor point of the range which is normally the beginning, 
    *  but will be the end if isReverse is set to true.
    *
    * @param[in] dirI3 the direction the particle is traveling.
    *
    * @param[in] isTau if set the lepton will be treated as a tau, otherwsie as a muon.
    *
    * @param[in] isReverse if true, it set pos as end point
    *  and calculate m.w.e or length in opposite direction of
    *  the track. If you want to know how far from a muon can
    *  reach to the detector, set pos as the entrance of icecube
    *  and set isReverse = true.
    *
    * @param[in] opt, scale
    * Used by EarthModelCalculator::GetLeptonRange() function.
    * Leave it as defaut unless you understand well about
    * the parameters.
    * 
    * @return distance[m]
    *
    */
   const double GetLeptonRangeInMeter(
                    double energy,
                    const  LeptonInjector::LI_Position &posI3,
                    const  LeptonInjector::LI_Direction &dirI3,
                    bool   isTau = false,
                    bool   isReverse = false,  
                    EarthModelCalculator::LeptonRangeOption 
                           opt = EarthModelCalculator::DEFAULT,
                    double scale = 1.0) const;

   /**
    * @brief A utility function to get depth for integration
    */
   const double FlatDepthCalculator(const LeptonInjector::LI_Position &frompos_CE,
                                    const LeptonInjector::LI_Position &topos_CE,
                                    double density,
                                    IntegType intg_type) const;
   
   /**
    * @brief A dummy variable used as a sink for ignored results.
    *
    * This variable's value should be considered undefined and it should never be read.
    */
   static bool ignored_bool;

   /**
    @brief path to data file
    */
   std::string  fPath_; 

   /**
    @brief name of EarthModel data file
    */
   std::vector<std::string>  fEarthModelStrings_;

   /**
    @brief name of Material Ratio Map data file
    */
   std::vector<std::string>  fMatRatioStrings_;

   /**
    @brief name of EarthModel data file
    */
   double fMohoBoundary_;           // [m], boundary between mantle and crust
   double fOutermostRockBoundary_;  // [m], boundary between rock and ice
   double fEarthAirBoundary_;       // [m], boundary between dense material and air
   double fAtmoRadius_;       // [m], Atmosphere radius

   double fDetDepth_;   // [m] depth of I3 origin from ice surface
   LeptonInjector::LI_Position fDetPos_;   // position of I3 origin in Earth-centered coordinate

   //-----------------------------------------
   // option for icecap at south pole
   // NoIce        : no ice option
   // IceSheet     : sheet of ice covers whole Earth (default)
   // SimpleIceCap : simple spherical ice cap 
   //

   EarthModelService::IceCapType fIceCapType_;
   std::string fIceCapTypeString_;

   double  fIceCapSimpleAngle_;  // [rad] open angle of simple icecap
   double  fIceCapSimpleRadius_; // [m] radius of shpere of simple icecap
   double  fIceCapSimpleZshift_; // [m] z-pos of icecap sphere

   // density data map  
   EarthParamMap    fEarthParams_;
   EarthParamMap    fIceParams_;

   /**
    * @brief map of materials and their WEIGHT ratio in unit volume.
    * used by Genie
    * format : map<MediumType, map<int(pdg nuclear code), double> > 
    * example :
    * [Ice]
    *    [1000080160][0.8881016]  // O
    *    [1000010010][0.1118983]  // H2
    */
   MatRatioMap fMatRatioMap_;

   /**
    * @brief map of light particles and their NUMBER ratio in unit volume
    * used by NuGen or may be by oscillation calculation 
    * format : map<MediumType, map<int(pdg code), double> > 
    * example :
    * [Ice]
    *    [2212][0.5555555555]  // proton
    *    [2112][0.4444444444]  // neutron
    *    [11][0.5555555555]    // electron, the weight is called Ye value
    * This map will be filled automatically when you set material file.
    */
   MatRatioMap fPNERatioMap_;

   /**
    * convert material pdg code to 
    * number of protons and neutrons
    */
    void GetAZ(int pdgcode, int& np, int& nn);
   
   /**
    * Computes the next point at which a track intersects the given material layer. 
    * This function exists as a helper for DistanceToNextBoundaryCrossing, and 
    * should probably not be used from other places. 
    *
    * @param posCE the starting point of the track in EarthCenter coordinate
    * @param dirCE the direction of the track in EarthCenter coordinate
    * @param ep    the layer to check for intersection
    * @param[out]  closestBoundary if the distance to ep is smaller than closestDist 
    *              ep will be copied to this variable
    * @param[out]  closestDist the closest boundary crossing yet found, which will be 
    *              updated if the crossing point for this boundary is closer
    * @param[out]  willExit whether the next crossing will involve the track leaving 
    *              this layer (ep)
    */
   bool CheckIntersectionWithLayer(
                const LeptonInjector::LI_Position& posCE,
                const LeptonInjector::LI_Direction& dirCE,
                EarthParamMap::const_iterator ep,
                EarthParamMap::const_iterator& closestBoundary,
                double& closestDist,
                bool& willExit) const;
   
   ///A helper structure for path integration of density
   struct densityPathIntegrand {

      ///The medium in which the integration should be done
      const EarthParam& medium;
      ///The base point for the path along which the integration is performed.
      ///Must be in earth-centered coordinates.
      LeptonInjector::LI_Position pos;
      ///The direction along which the integration is performed.
      ///Must be in earth-centered coordinates.
      LeptonInjector::LI_Direction dir;

      IntegType intg_type;
      
      densityPathIntegrand(const EarthParam& e, const LeptonInjector::LI_Position& p, const LeptonInjector::LI_Direction& d, IntegType integ_type):
      medium(e),pos(p),dir(d),intg_type(integ_type) {}
      
      ///Evaluates the density at the appropriate point along the parameterized track
      // returns in cgs unit
      double operator()(double dist) const {
         LeptonInjector::LI_Position curpos = pos + dist*dir;
         double density = GetEarthDensityInCGS(medium,curpos);

         if (intg_type == PATH || intg_type == RADIUS) {
            return density;  // [g/cm^3]
         }

         double r_cm = curpos.Magnitude() * LeptonInjector::Constants::m / LeptonInjector::Constants::cm;
         if (intg_type == SPHERE) {
            // calculate sphere * density [g/cm]
            // we assume dist is small ehough 
            return 4 * r_cm * r_cm * LeptonInjector::Constants::pi * density;

         } else if (intg_type == CIRCLE) {
            // calculate circle line * density  [g/cm^2]
            // we assume dist is small ehough 
            return 2 * r_cm * LeptonInjector::Constants::pi * density;
         }
         throw("wrong intg type " +std::to_string(static_cast<int>(intg_type)));
         return -1;
      }

   };

};

}

#endif
