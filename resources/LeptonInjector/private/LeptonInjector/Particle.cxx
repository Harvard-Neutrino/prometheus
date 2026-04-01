#include <math.h>
#include <stdexcept> // adds sqrt, power functions
#include <LeptonInjector/Particle.h>
#include <assert.h>
#include <map>

namespace LeptonInjector{
    Particle::Particle(void){
        // Just sit it at the origin 
        
        // note that
        // static_cast<int32_t>(ParticleType::EMinus) == 11
        type        = ParticleType::unknown;
        
        // instantiate with minimum energy
        energy      = 0.0;
        direction   = std::make_pair( 0.0, 0.0) ;
		for (uint8_t var = 0; var<3; var++){
			position[var] = 0.0;
		}
    }
    
    // constructor for specific type
    Particle::Particle( ParticleType type ){
        this->type  = type;

        // energy will instead be built using particle's mass
        energy      = this->GetMass();
        direction   = std::make_pair(0.0, 0.0);
		for (uint8_t var = 0; var<3; var++){
			position[var] = 0.0;
		}
    }



	// returns name for particle of known type. 
	// If this code is to be expanded, this should really be modified to use the boost preprocessor libraries
	// atm, only implemented for the particles relevant to LeptonInjector 
	std::string Particle::GetTypeString(){

		// there is a way to do this better with boost preprocessor libraries, but I think that's a little unnecessary given the scope of what LI does. 

		// this just casts the particle type to its pdg code, and uses a switch to grab the name
		switch( static_cast<int32_t>(this->type) ){
			case 0: return("Unknown"); break;
			case 22: return("Gamma"); break;
			case 11: return("EMinus"); break;
			case -11: return("EPlus"); break;
			case 13: return("MuMinus"); break;
			case -13: return("MuPlus"); break;
			case 15: return("TauMinus"); break;
			case -15: return("TauPlus"); break;
			case 12: return("NuE"); break;
			case -12: return("NuEBar"); break;
			case 14: return("NuMu"); break;
			case -14: return("NuMuBar"); break;
			case 16: return("NuTau"); break;
			case -16: return("NuTauBar"); break;
			case -2000001006: return("Hadrons"); break;
			default: return("Unsupported"); break;
		}

	}

    bool Particle::HasMass(){
        // return the negation of the bool that (particle is massless)
        return(!( this->type == ParticleType::Gamma || 
               this->type == ParticleType::NuE   || this->type==ParticleType::NuEBar   ||
               this->type == ParticleType::NuMu  || this->type==ParticleType::NuMuBar  ||
               this->type == ParticleType::NuTau || this->type==ParticleType::NuTauBar ||
               this->type == ParticleType::Hadrons) );
    }

    // only implemented for the charged leptons to stay within scope
    double Particle::GetMass(){
        switch(this->type){
            case ParticleType::EPlus:
                return( Constants::electronMass );
                break;
            case ParticleType::EMinus:
                return( Constants::electronMass );
                break;
            case ParticleType::MuPlus:
                return( Constants::muonMass );
                break;
            case ParticleType::MuMinus:
                return( Constants::muonMass );
                break;
            case ParticleType::TauPlus:
                return( Constants::tauMass );
                break;
            case ParticleType::TauMinus:
                return( Constants::tauMass );
                break;
			case ParticleType::PPlus:
				return( Constants::protonMass );
				break;
			case ParticleType::Neutron:
				return( Constants::neutronMass);
            default:
                return(0.0);
        }
    }



    // Helper functions for dealing with particle types  

    // returns true if a particle is a Lepton. False if not
    bool isLepton(Particle::ParticleType p){
		return(p==Particle::ParticleType::EMinus   || p==Particle::ParticleType::EPlus ||
			   p==Particle::ParticleType::MuMinus  || p==Particle::ParticleType::MuPlus ||
			   p==Particle::ParticleType::TauMinus || p==Particle::ParticleType::TauPlus ||
			   p==Particle::ParticleType::NuE      || p==Particle::ParticleType::NuEBar ||
			   p==Particle::ParticleType::NuMu     || p==Particle::ParticleType::NuMuBar ||
			   p==Particle::ParticleType::NuTau    || p==Particle::ParticleType::NuTauBar);
	}
	
    // returns true if the particle is either
    //        a charged lepton 
    //   (OR) a "hadrons" particle
	bool isCharged(Particle::ParticleType p){
		if( !(isLepton(p) || p==Particle::ParticleType::Hadrons) ){
			throw std::runtime_error("You should only be using Leptons or Hadrons!");
		}
		
		// keeps this within scope. Shouldn't be getting some other kind of charged particle
		return(p==Particle::ParticleType::EMinus   || p==Particle::ParticleType::EPlus ||
			   p==Particle::ParticleType::MuMinus  || p==Particle::ParticleType::MuPlus ||
			   p==Particle::ParticleType::TauMinus || p==Particle::ParticleType::TauPlus ||
			   p==Particle::ParticleType::Hadrons);
	}


    // returns string of particle's name
	std::string particleName(Particle::ParticleType p){
		return(Particle(p).GetTypeString());
	}
	

    // gets the mass of a particle for a given type
	double particleMass(Particle::ParticleType type){
		Particle p(type);
		if(!p.HasMass()){
			return(0);
		}
		return(p.GetMass());
	}
	
    // Uses a particle's type (mass) and total energy to calculate kinetic energy
	double kineticEnergy(Particle::ParticleType type, double totalEnergy){
		double mass=particleMass(type);
		if(totalEnergy<mass){
			return(0.);
		}
		return(sqrt(totalEnergy*totalEnergy-mass*mass));
	}
	
    // uses the particle type and kinetic energy to calculate the speed of the particle
    // relies on the constants! 
	double particleSpeed(Particle::ParticleType type, double kineticEnergy){
		Particle p=Particle(type);
		if(!p.HasMass()){
			return(Constants::c);
		}
		double mass=p.GetMass();
		if(kineticEnergy<0){
			return(0.);
		}

        // IF mass>0 THEN return mass/(stuff) ... ELSE return 0
		double r=(mass>0 ? mass/(kineticEnergy+mass) : 0.);
		return(Constants::c*sqrt(1-r*r));
	}
	
	Particle::ParticleShape decideShape(Particle::ParticleType t){
		switch(t){
			case Particle::ParticleType::MuMinus:  case Particle::ParticleType::MuPlus:
			case Particle::ParticleType::TauMinus: case Particle::ParticleType::TauPlus:
			case Particle::ParticleType::NuE:      case Particle::ParticleType::NuEBar:
			case Particle::ParticleType::NuMu:     case Particle::ParticleType::NuMuBar:
			case Particle::ParticleType::NuTau:    case Particle::ParticleType::NuTauBar:
				return(Particle::ParticleShape::MCTrack);
			case Particle::ParticleType::EMinus: case Particle::ParticleType::EPlus:
			case Particle::ParticleType::Hadrons:
				return(Particle::ParticleShape::Cascade);
			case Particle::ParticleType::unknown:
				return(Particle::ParticleShape::unknown);
			default:
                throw "BadShape"; // this replaces the previous fatal log
//				log_fatal_stream("Unable to decide shape for unexpected particle type: " << particleName(t));
		}
	}

	uint8_t getInteraction( Particle::ParticleType final_1 , Particle::ParticleType final_2){
		// check for GR
		if ((final_1==Particle::ParticleType::EMinus && final_2 == Particle::ParticleType::NuEBar)||
		(final_1==Particle::ParticleType::MuMinus && final_2 == Particle::ParticleType::NuMuBar)||
		(final_1==Particle::ParticleType::TauMinus && final_2 == Particle::ParticleType::NuTauBar)||
		(final_2==Particle::ParticleType::EMinus && final_1 == Particle::ParticleType::NuEBar)||
		(final_2==Particle::ParticleType::MuMinus && final_1 == Particle::ParticleType::NuMuBar)||
		(final_2==Particle::ParticleType::TauMinus && final_1 == Particle::ParticleType::NuTauBar)||
		(final_1==Particle::ParticleType::Hadrons && final_2 == Particle::ParticleType::Hadrons)){
			return( 2 ); // glashow resonance 
		}else if( ((final_2==Particle::ParticleType::Hadrons) and (
			final_1==Particle::Particle::EPlus || final_1==Particle::Particle::EMinus ||
			final_1==Particle::Particle::MuPlus || final_1==Particle::Particle::MuMinus ||
			final_1==Particle::Particle::TauPlus || final_1==Particle::Particle::TauMinus)) or
		    ((final_1==Particle::ParticleType::Hadrons) and (
			final_2==Particle::Particle::EPlus || final_2==Particle::Particle::EMinus ||
			final_2==Particle::Particle::MuPlus || final_2==Particle::Particle::MuMinus ||
			final_2==Particle::Particle::TauPlus || final_2==Particle::Particle::TauMinus ))){
			return( 0 ); // charged current
		}else if( ((final_2==Particle::ParticleType::Hadrons) and (
			final_1==Particle::Particle::NuEBar || final_1==Particle::Particle::NuE ||
			final_1==Particle::Particle::NuMuBar || final_1==Particle::Particle::NuMu ||
			final_1==Particle::Particle::NuTauBar || final_1==Particle::Particle::NuTau )) or
		    ((final_1==Particle::ParticleType::Hadrons) and (
			final_2==Particle::Particle::NuEBar || final_2==Particle::Particle::NuE ||
			final_2==Particle::Particle::NuMuBar || final_2==Particle::Particle::NuMu ||
			final_2==Particle::Particle::NuTauBar || final_2==Particle::Particle::NuTau ))){
			return( 1 ); // neutral current
		}
		
		throw std::runtime_error("Interaction type not recognized");
	}

    // This function returns the primary particle type given the final state particles
    // returns a particle type object    
	Particle::ParticleType deduceInitialType(Particle::ParticleType pType1, Particle::ParticleType pType2){
		//only accept certain particle types in general
		if(!isLepton(pType1) && pType1!=Particle::ParticleType::Hadrons)
            throw std::runtime_error("BadParticle"); 
		if(!isLepton(pType2) && pType2!=Particle::ParticleType::Hadrons)
            throw std::runtime_error("BadParticle");
		
		bool c1=isCharged(pType1);
		bool c2=isCharged(pType2);
		bool l1=isLepton(pType1);
		bool l2=isLepton(pType2);
		
		//at least one particle should be charged
		if(!c1 && !c2)
			throw std::runtime_error("Final state should have at least one charged particle");
		//first particle is charged, second is not
		if(c1 && !c2){
			//valid cases are charged lepton + matching antineutrino for GR
			if(l1){
				//!c2 => pType2 is a neutrino
				if(!((pType1==Particle::ParticleType::EMinus   && pType2==Particle::ParticleType::NuEBar) ||
					 (pType1==Particle::ParticleType::EPlus    && pType2==Particle::ParticleType::NuE) ||
					 (pType1==Particle::ParticleType::MuMinus  && pType2==Particle::ParticleType::NuMuBar) ||
					 (pType1==Particle::ParticleType::MuPlus   && pType2==Particle::ParticleType::NuMu) ||
					 (pType1==Particle::ParticleType::TauMinus && pType2==Particle::ParticleType::NuTauBar) ||
					 (pType1==Particle::ParticleType::TauPlus  && pType2==Particle::ParticleType::NuTau)))
                     throw std::runtime_error("Final states with a charged lepton must have an anti-matching neutrino.");
				return(Particle::ParticleType::NuEBar);
			}
            throw std::runtime_error("BadFinal");
		}
		
		//first particle is neutral, second is charged
		if(!c1 && c2){
			if(l1 && pType2==Particle::ParticleType::Hadrons){
				//particle 1 is a neutral lepton, so it must be a neutrino
				return(pType1); //the incoming neutrino type is the same as the outgoing
			}
            throw std::runtime_error("BadFinal");
		}
		
		//have two charged particles
		if(c1 && c2){
			//no two charged lepton states
			if(l1 && l2)
                throw std::runtime_error("BadFinal");
			//lepton should be given first
			if(!l1 && l2)
                throw std::runtime_error("BadFinal");
			if(l1 && !l2){ //valid: charged lepton + Hadrons for CC
				switch(pType1){
					case Particle::ParticleType::EMinus: return(Particle::ParticleType::NuE);
					case Particle::ParticleType::EPlus: return(Particle::ParticleType::NuEBar);
					case Particle::ParticleType::MuMinus: return(Particle::ParticleType::NuMu);
					case Particle::ParticleType::MuPlus: return(Particle::ParticleType::NuMuBar);
					case Particle::ParticleType::TauMinus: return(Particle::ParticleType::NuTau);
					case Particle::ParticleType::TauPlus: return(Particle::ParticleType::NuTauBar);
					default: assert(false && "This point should be unreachable");
				}
			}
			if(!l1 && !l2){ //valid: two hadrons (for GR)
				return(Particle::ParticleType::NuEBar);
			}
		}
        throw std::runtime_error("You must be a wizard: this point should be unreachable");
	}


    // Particle-based exceptions:

} // end namespace LI_Particle
