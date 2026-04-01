#ifndef LI_COORDS
#define LI_COORDS

#include <array>
#include <cstdint> // used for many of the constructors, and for the position
#include <math.h> //sqrt, sin, cos, pow
#include <exception> //allows throwing the out_of_range exception
#include <limits> //numeric limits
#include <iostream>

// Ben Smithers
// benjamin.smithers@mavs.uta.edu

// Implements tools and classes for working with positions and directions
// Angles presumed to be in radians. 

namespace LeptonInjector {

	// number of linearly independent bases are needed to span this physical space
	// do not cchange
	static const int n_dimensions = 3;

	// Creating a "LI_Position" to mimic the I3_Position object
	// this is more accurately described as a vector in cartesian coordinates
	//
	// Assmued to be in cartesian for defined operations
	class LI_Position{
		public: 
			LI_Position();

			LI_Position(double x, double y, double z);
			LI_Position(const LI_Position& old_one);
			LI_Position(std::array<double, n_dimensions> pos);

			double at(uint8_t component) const;
			//double Magnitude(void);
			double Magnitude(void) const;

			// added for historical reasons
			double GetZ() const;
			double GetY() const;
			double GetX() const;

			// added for that last beit of needed functionality 
			// I don't like adding these, since it violates that whole trust of the "const" above.
			void SetZ(double amt);
			void SetY(double amt);
			void SetX(double amt);

		private:
			std::array<double, n_dimensions> position;
	};

	// Creating a "LI_Direction" to mimic the I3_Direction object
	// this should behave in the same exact way, at least within the scope of this project 
	class LI_Direction{
		public:
			LI_Direction();

			LI_Direction( double theta, double phi);
			LI_Direction( std::array<double, 2> dir);
			LI_Direction( std::pair<double, double> dir);
			LI_Direction( const LI_Direction& old_one);
			LI_Direction( const LI_Position& vec );  // get the direction of a vector 

			double GetZ() const;
			double GetY() const;
			double GetX() const;
			
			double zenith;
			double azimuth;
	};
	

	// these functions rotate a 3-vector about some axis
	// assumes vector is in cartesian coordinates
	LI_Position RotateY(LI_Position vector, double angle);
	LI_Position RotateX(LI_Position vector, double angle);
	LI_Position RotateZ(LI_Position vector, double angle);

	// defines scalar multiplication and its conjugation 
	LI_Position operator * (const LI_Position& point, double scalar);
	LI_Position operator * (double scalar,const LI_Position& point);

	// Defines multiplication of a unit-vector (or direction) by a scalar
	LI_Position operator * (const LI_Direction&  dir, double scalar);
	LI_Position operator * (double scalar,const LI_Direction& dir);

	// returns the length of the projection of a vector along a direction
	double operator * (const LI_Position&  pos,const LI_Direction&  dir);
	double operator * (const LI_Direction&  dir,const LI_Position&  pos);

	// Define the inner product of two positions by treating them as vectors
	double operator * (const LI_Position& vec1,const LI_Position& vec2);

	// Define addition, subtraction
	LI_Position operator + (const LI_Position&  pos1,const LI_Position& pos2);
	LI_Position operator - (const LI_Position&  pos1,const LI_Position& pos2);
	LI_Position& operator += (LI_Position& one, const LI_Position& two);
	LI_Position& operator -= (LI_Position& one, const LI_Position& two);
	LI_Direction operator - (LI_Direction obj);

	// Define string-casting of a position. Used often for error messages 
	std::ostream & operator << (std::ostream& out, const LI_Position &dir);

	bool operator == (LI_Position const & one, LI_Position const & two);

	// Takes a direciton, rotates it about the Y-axis by the zenith amount, then about the Z axis by the azimuth amount
	LI_Direction rotateRelative(const LI_Direction& base, double zenith, double azimuth);

    std::tuple<LI_Position, LI_Position> computeCylinderIntersections(const LI_Position& pos, const LI_Direction& dir, double radius, double z_min, double z_max);


} // end namespace LeptonInjector

#endif
