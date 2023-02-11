float xrnd(){
  unsigned int rnd;
  do rnd=rand_r(&sv); while(rnd==0);
  const float a=1.0f/(1ll+RAND_MAX);
  return a*rnd;
}

#if defined(MKOW) || defined(XLIB)
float grnd(){  // gaussian distribution
  return sqrtf(-2*logf(xrnd()))*sinf(2*FPI*xrnd());
}
#endif

static const float ppm=2450.08;     // photons per meter
static const float rho=0.9216;      // density of ice [mwe]
static const float m0=0.105658389;  // muon rest mass [GeV]

photon p;

#ifdef XLIB
float yield(float N, char type){  // LED light
  p.l=-1;
#ifdef ANGW
  p.f=1;
#endif
#ifdef LONG
  p.a=0, p.b=0;
#endif
  return d.eff*N;
}
#endif

#ifdef CWLR
float yield(float E, int type){  // cascades
  float f=1.0f;
  float logE=logf(max(m0, type<0?10:E));

  const float Lrad=0.39652*0.910f/rho;
  const float em=5.321*0.910f/rho;  // 0.910 density used in simulation

  /**
   * a,b describe the longitudinal profile of cascades.
   * The energy dependence of a is given by p1+p2*logE and b is constant.
   * Add comment about validity of parameterization for low-energy cascades.
   *
   * Total light yield of cascades:
   * For e-,e+,gamma the light yield is given by p*simulated_density/ice_density.
   * For hadrons the light yield is scaled down by Fi where i denotes the particle type.
   * The parameterizations have been derived from fits in an energy range
   * from 30 GeV to 10 TeV. Below approximately 10 GeV the F is not described
   * by F = 1-(E/E0)^(-m)*(1-f0) and is therefore a rather crude approximation.
   * Fluctuations have been parameterized with sigma/F = rms0*ln(E[GeV])^(-gamma).
   * For antiprotons the annihilation cross section has to be taken into account
   * at small energies. At the moment F for protons is used but with the fluctuations 
   * of antiprotons.
   *
   * Reference: icecube/201210001 (for hadrons the values are recalculated)
   * The type are as following:
   * type  particle
   * 1 standard em (same as e-) is default if no other type is given
   * 2 e-
   * 3 e+
   * 4 gamma
   * 101 standard hadron (same as pi+) 
   * 102 pi+
   * 103 pi-
   * 104 kaon0L
   * 105 proton
   * 106 neutron
   * 107 anti_proton Use F for protons because parameterization fails for small
   *     energies due to annihilation cross section. However sigma_F is used from
   *     anti_proton parameterization.
   **/

  p.l=0, p.f=0;

  if(type>100){
    float E0, m, f0, rms0, gamma;

    switch(type){
    default:
    case 101: // standard hadron (same as pi+)
    case 102: // pi+
      p.a=1.58357292f+0.41886807f*logE, p.b=Lrad/0.33833116f;
      E0=0.18791678f;
      m =0.16267529f;
      f0=0.30974123f;
      rms0 =0.95899551f;
      gamma=1.35589541f;
      break;

    case 103: // pi-
      p.a=1.69176636f+0.40803489f*logE, p.b=Lrad/0.34108075f;
      E0=0.19826506f;
      m =0.16218006f;
      f0=0.31859323f;
      rms0 =0.94033488f;
      gamma=1.35070162f;
      break;

    case 104: // kaon0L
      p.a=1.95948974f+0.34934666f*logE, p.b=Lrad/0.34535151f;
      E0=0.21687243f;
      m =0.16861530f;
      f0=0.27724987f;
      rms0 =1.00318874f;
      gamma=1.37528605f;
      break;

    case 105: // proton
      p.a=1.47495778f+0.40450398f*logE, p.b=Lrad/0.35226706f;
      E0=0.29579368f;
      m =0.19373018f;
      f0=0.02455403f;
      rms0 =1.01619344f;
      gamma=1.45477346f;
      break;

    case 106: // neutron
      p.a=1.57739060f+0.40631102f*logE, p.b=Lrad/0.35269455f;
      E0=0.66725124f;
      m =0.19263595f;
      f0=0.17559033f;
      rms0 =1.01414337f;
      gamma=1.45086895f;
      break;

    case 107: // anti_proton
      p.a=1.92249171f+0.33701751f*logE, p.b=Lrad/0.34969748f;
      E0=0.29579368f;
      m =0.19373018f;
      f0=0.02455403f;
      rms0 =1.01094637f;
      gamma=1.50438415f;
      break;
    }

    {
      float e=max(2.71828183f, E);
      float F=1-powf(e/E0, -m)*(1-f0);
      float dF=F*rms0*powf(logf(e), -gamma);
      do f=F+dF*grnd(); while(f<0 || 1.1<f);
    }
  }
  else{
    switch(type){
    default:
    case 1:   // em shower
    case 2:   // e-
      p.a=2.01849f+0.63176f*logE, p.b=Lrad/0.63207f;
      break;

    case 3:   // e+
      p.a=2.00035f+0.63190f*logE, p.b=Lrad/0.63008f;
      break;

    case 4:   // gamma
      p.a=2.83923f+0.58209f*logE, p.b=Lrad/0.64526f;
      break;
    }
  }

  float nph=f*em;
  return d.eff*ppm*nph*E;
}

float yield(float E, float dr){  // bare muon
  float logE=logf(max(m0, E));
  float extr=1+max(0.0f, 0.1880f+0.0206f*logE);
  float nph=dr>0?dr*extr:0;
  p.l=dr;
  p.f=1/extr;
  p.a=0, p.b=0;
  return d.eff*ppm*nph;
}

#else
float yield(float E, int type){
  p.l=0;
#ifdef ANGW
  p.f=0;
#endif
#ifdef LONG
  float logE=logf(max(m0, type<0?10:E));
  if(type>100){  // hardonic shower
    const float Lrad=0.95/rho;
    p.a=1.49f+0.359f*logE, p.b=Lrad/0.772f;
  }
  else{  // em shower
    const float Lrad=0.358/rho;
    p.a=2.03f+0.604f*logE, p.b=Lrad/0.633f;
  }
#endif

  float nph;
#ifdef MKOW
  const float em=5.21*0.924f/rho;

  float f=1.0f;
  if(type>100){
    const float E0=0.399;
    const float m=0.130;
    const float f0=0.467;
    const float rms0=0.379;
    const float gamma=1.160;

    float e=max(10.0f, E);
    float F=1-powf(e/E0, -m)*(1-f0);
    float dF=F*rms0*powf(log10f(e), -gamma);
    do f=F+dF*grnd(); while(f<0 || 1.1<f);
  }
  nph=f*em;
#else
  const float em=4.889*0.894f/rho;  // em shower
  const float hr=4.076*0.860f/rho;  // hdr [m/GeV]
  nph=type>100?hr:em;
#endif
  return d.eff*ppm*nph*E;
}

float yield(float E, float dr){     // bare muon
  float logE=logf(max(m0, E));
  float extr=1+max(0.0f, 0.1720f+0.0324f*logE);
  float nph=dr>0?dr*extr:0;
  p.l=dr;
#ifdef ANGW
  p.f=1/extr;
#endif
#ifdef LONG
  p.a=0, p.b=0;
#endif
  return d.eff*ppm*nph;
}
#endif

#ifdef XLIB
struct mcid:pair<int,unsigned long long>{
  int frame;
};

struct ihit{
  ikey omkey;
  mcid track;
  float time;
} tmph;

deque<mcid> flnz;
vector<ihit> hitz;
#else
deque<string> flnz;
#endif

unsigned int flnb=0, flne=0;
#ifdef XCPU
unsigned int & flnd = flne;
#else
unsigned int flnd=0;
#endif

#ifndef RAND
int sign(float x){ return x<0?-1:x>0?1:0; }
#endif

int hcmp(const void *a, const void *b){
  hit & ah = * (hit *) a;
  hit & bh = * (hit *) b;
#ifdef RAND
  return (int) ( ah.n - bh.n );
#else
  return ah.n!=bh.n ? (int)(ah.n-bh.n) : ah.i!=bh.i ? (int)(ah.i-bh.i) : ah.t!=bh.t ? sign(ah.t-bh.t) : sign(ah.z-bh.z);
#endif
}

void print(){
#ifdef RAND
  if((int) ( flnb - flnd ) < 0)
#endif
    qsort(q.hits, d.hidx, sizeof(hit), hcmp);

  for(unsigned int i=0; i<d.hidx; i++){
    hit & h = q.hits[i];
    if((int)(h.n-flnb)<0 || (int)(h.n-flnd)>0){
      cerr<<"Internal Error: "<<h.n<<" is not between "<<flnb<<" "<<flnd<<" "<<flne<<endl;
      continue;
    }

    for(; (int) ( flnb - h.n ) < 0; flnb++){
#ifdef XLIB
      tmph.track=flnz.front();
#else
      cout<<flnz.front()<<endl;
#endif
      flnz.pop_front();
    }

    name n=q.names[h.i];
    float rde=n.rde, f;
    if(rdef){
      float r=q.rde[h.z];
      if(rde>1) f=r>1?1:r;
      else f=r>1?1/r:1;
    }
    else f=rde/rmax;

    if(n.hv>0 && xrnd()<f){
#ifdef XLIB
      tmph.omkey=n;
      tmph.time=h.t;
      hitz.push_back(tmph);
#else
      printf("HIT %d %d %f %f\n", n.str, n.dom, h.t, h.z);
#endif
    }
  }

  for(; (int) ( flnb - flnd ) < 0; flnb++){
#ifdef XLIB
    tmph.track=flnz.front();
#else
    cout<<flnz.front()<<endl;
#endif
    flnz.pop_front();
  }
}

void output(){
  kernel(pn*OVER); pn=0;

#ifndef XCPU
  flnd=flne;
#endif
}

unsigned long long f2ki, f2kn;

template <class T> void addp(float rx, float ry, float rz, float t, float E, T xt){
  p.r.w=t; p.r.x=rx; p.r.y=ry; p.r.z=rz;
  unsigned long long num=llroundf(yield(E, xt));
  for(f2kn+=num; f2ki<f2kn; f2ki+=ovr){
    q.pz[pn++]=p; if(pn==pmxo) output();
  }
}

#ifdef XLIB
template void addp(float, float, float, float, float, int);
template void addp(float, float, float, float, float, char);
template void addp(float, float, float, float, float, float);
#endif

void eini(){
  f2ki=0;
  f2kn=0;
}

void finc(){
  flne++;
  if((int) ( flne - flnb ) < 0){ cerr<<"Error: too many event segments"<<endl; exit(3); }
}

void eout(){
  output();
#ifndef XCPU
  output();
#endif
}

#ifdef XLIB
void efin(){
  hitz.erase(hitz.begin(), hitz.end());
}

void sett(float nx, float ny, float nz, pair<int,unsigned long long> id, int frame){
  mcid ID; ID.first=id.first; ID.second=id.second; ID.frame=frame;
  flnz.push_back(ID); finc();
  p.q=flne; p.n.x=nx; p.n.y=ny; p.n.z=nz;
}
#else

void f2k(){
  ini(0);

  string in;
  while(getline(cin, in)){
    flnz.push_back(in); finc();
    char name[32];
    int gens, igen;
    float x, y, z, th, ph, l, E, t;
    const char * str = "TR %d %d %31s %f %f %f %f %f %f %f %f";

    if(sscanf(in.c_str(), "%31s", name)==1){ if(strcmp(name, "EM")==0) eini(); }
    if(sscanf(in.c_str(), str, &gens, &igen, name, &x, &y, &z, &th, &ph, &l, &E, &t)==11){
      th=180-th; ph=ph<180?ph+180:ph-180;
      th*=FPI/180; ph*=FPI/180;
      float costh=cosf(th), sinth=sinf(th), cosph=cosf(ph), sinph=sinf(ph);
      p.q=flne; p.n.x=sinth*cosph; p.n.y=sinth*sinph; p.n.z=costh;
      if(0==strcmp(name, "amu+") || 0==strcmp(name, "amu-") || 0==strcmp(name, "amu")) addp(x, y, z, t, E, l);
      else{
	int type=0;
	if(0==strcmp(name, "delta") || 0==strcmp(name, "brems") ||
	   0==strcmp(name, "epair") || 0==strcmp(name, "e")) type=1;
	else if(0==strcmp(name, "e-")) type=2;
	else if(0==strcmp(name, "e+")) type=3;
	else if(0==strcmp(name, "munu") || 0==strcmp(name, "hadr")) type=101;
	if(type>0) addp(x, y, z, t, E, type);
      }
    }
  }
  eout();

  fin();
}
#endif
