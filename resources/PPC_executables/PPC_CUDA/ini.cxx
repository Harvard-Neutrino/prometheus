#define OFLA         // omit the flasher DOM
#define ROMB         // use rhomb cells aligned with the array
#define ASENS        // enable angular sensitivity
#define RAND         // disable for deterministic results

#define TILT         // enable tilted ice layers
#define ANIZ         // enable anisotropic ice
//#define HOLE       // enable direct hole ice simulation

#define MKOW         // photon yield parametrizations by M. Kowalski
#define ANGW         // smear cherenkov cone due to shower development
#define LONG         // simulate longitudinal cascade development
#define CWLR         // new parameterizations by C. Wiebusch and L. Raedel
                     // requires that MKOW, ANGW, and LONG are all defined

#ifdef ASENS
#define ANUM 11      // number of coefficients in the angular sensitivity curve
#endif

#ifdef TILT
#define LMAX 6       // number of dust loggers
#define LYRS 170     // number of depth points
#endif

#ifdef ROMB
#define DIR1 9.3
#define DIR2 129.3

#define CX   21
#define CY   19
#define NSTR 94
#else
#define CX   13
#define CY   13
#define NSTR 94
#endif

#ifdef XCPU
#define OVER 1
#define NBLK 1
#define NTHR 512
#else
#define OVER 10      // size of photon bunches along the muon track
#endif

//#define TALL       // enable faster 2-stage processing, takes more memory
#define HQUO   16    // save at most photons/HQUO hits
#define NPHO   1024  // maximum number of photons propagated by one thread

#define WNUM   32    // number of wavelength slices
#define MAXLYS 180   // maximum number of ice layers
#define MAXGEO 5200  // maximum number of OMs
#define MAXRND 131072   // max. number of random number multipliers

#define XXX 1.e-5f
#define FPI 3.141592653589793f
#define OMR 0.16510f // DOM radius [m]

const float fcv=FPI/180.f;
static unsigned int ovr=OVER;

struct DOM{
  float r[3];
};

struct ikey{
  int str, dom;

  bool isinice() const{
    return str>0 && dom>=1 && dom<=60;
  }

  bool operator< (const ikey & rhs) const {
    return str == rhs.str ? dom < rhs.dom : str < rhs.str;
  }
};

struct OM:DOM,ikey{};
vector<OM> i3oms;

struct name:ikey{
  float rde, hv;

  name(){}
  name(ikey k, float r, float h):ikey(k){
    rde=r; hv=h;
  }
};

map<ikey, float> rdes, hvs;

bool rdef=false;
float rmax=1;

struct hit{
  unsigned int i;
  float t;
  unsigned int n;
  float z;
  float pth, pph, dth, dph;
};

#ifdef XCPU
struct float2{
  float x, y;
};

struct float3:float2{
  float z;
};

struct float4:float3{
  float w;
};
#endif

struct pbuf{
  float4 r;        // location, time
  float3 n;        // direction
  unsigned int q;  // track segment
};

struct photon:pbuf{
  float l;         // track length
#ifdef ANGW
  float f;         // fraction of light from muon alone (without cascades)
#endif
#ifdef LONG
  float a, b;      // longitudinal development parametrization coefficients
#endif
};

struct ices{
  float wvl;             // wavelength of this block
  float ocm;             // 1 / speed of light in medium
  float coschr, sinchr;  // cos and sin of the cherenkov angle
  struct{
    float abs;           // absorption
    float sca;           // scattering
  } z [MAXLYS];
};

struct line{
  short n, max;
  float x, y, r;
  float h, d;
  float dl, dh;
};

struct datz{
  ices w[WNUM];
  unsigned int rm[MAXRND];
  unsigned long long rs[MAXRND];
} z;

struct dats{
  unsigned int hidx;

#ifndef XCPU
  unsigned int tn, tx;  // kernel time clocks
  unsigned int ab;      // if TOT was abnormal
  unsigned int mp;      // kernel block counter
  short bmp[4];         // list of 4 faulty MPs
#endif
  short blockIdx, gridDim;  // bad/current MP; number of MPs

  int type;   // 0=cascade/1=flasher/2=flasher 45/3=laser up/4=laser down
  float r[3]; // flasher/laser coordinates
  float ka, up;    // 2d-gaussian rms and zenith of cone

  unsigned int hnum;    // size of hits buffer
  int size;   // size of kurt table
  int rsize;  // count of multipliers
  int gsize;  // count of initialized OMs

  float dh, hdh, rdh, hmin; // step, step/2, 1/step, and min depth

  float ocv;  // 1 / speed of light in vacuum
  float sf;   // scattering function: 0=HG; 1=SAM
  float g, g2, gr; // g=<cos(scattering angle)>, g2=g*g and gr=(1-g)/(1+g)
  float R, R2, zR; // DOM radius, radius^2, and inverse "oversize" scaling factor

#ifdef HOLE
  float hr, hr2, hs, ha; // hole ice radius, radius^2, effective scattering and absorption coefficients
  float SF, G, G2, GR;   // hole ice sf, g, g2, gr
#endif

  int cn[2];
  float cl[2], crst[2];

  unsigned char is[CX][CY];
  unsigned char ls[NSTR];
  line sc[NSTR];
  float rx;

  float fldr; // horizontal direction of the flasher led #1
  float eff;  // OM efficiency correction

#ifdef ASENS
  float mas;  // maximum angular sensitivity
  float s[ANUM]; // ang. sens. coefficients
#endif

#ifdef ROMB
  float cb[2][2];
#endif
#ifdef TILT
  int lnum, lpts, l0;
  float lmin, lrdz, r0;
  float lnx, lny;
  float lr[LMAX];
  float lp[LMAX][LYRS];
#endif

  short fla;
#ifdef ANIZ
  short k;          // ice anisotropy: 0: no, 1: yes
  float k1, k2, kz; // ice anisotropy parameters
  float azx, azy;   // ice anisotropy direction
#endif

  datz * z;
  hit * hits;
  photon * pz;
#ifdef TALL
  pbuf * bf;
#endif
} d;

struct doms{
  DOM oms[MAXGEO];
  name names[MAXGEO];
  map<float, float> rde;

  hit * hits;
  photon * pz;
} q;

unsigned char sname(int n){
  name & s = q.names[n];
  return s.str>78&&s.dom>10?s.str+10:s.str;
}

static const float zoff=1948.07;
unsigned int sv=0;

void rs_ini(){
  union{
    unsigned long long da;
    struct{
      unsigned int ax;
      unsigned int dx;
    };
  } s;

  s.ax=362436069;
  s.dx=1234567;

  s.ax+=sv;

  for(int i=0; i<d.rsize; i++) z.rs[i]=s.da;
}

struct ini{
  float ctr(line & s, int m){
#ifdef ROMB
    return d.cb[0][m]*s.x+d.cb[1][m]*s.y;
#else
    return m==0?s.x:s.y;
#endif
  }

  void set(){
    {
      d.hidx=0; d.type=0; for(int m=0; m<3; m++) d.r[m]=0; d.ka=0, d.up=0; d.z=&z;
    }

    string dir("");
    {
      char * env = getenv("PPCTABLESDIR");
      if(env!=NULL) dir=string(env)+"/";
      cerr<<"Configuring in \""<<dir<<"\""<<endl;
    }

    {
      ifstream inFile((dir+"cfg.txt").c_str(), ifstream::in);
      if(!inFile.fail()){
	string in;
	float aux;
	vector<float> v;
	while(getline(inFile, in)) if(sscanf(in.c_str(), "%f", &aux)==1) v.push_back(aux);

	if(v.size()>=4){
	  int xR=lroundf(v[0]); d.zR=1.f/xR, ovr*=xR*xR, d.R=OMR*xR; d.R2=d.R*d.R;
	  d.eff=v[1], d.sf=v[2], d.g=v[3]; d.g2=d.g*d.g; d.gr=(1-d.g)/(1+d.g);
	  cerr<<"Configured: xR="<<xR<<" eff="<<d.eff<<" sf="<<d.sf<<" g="<<d.g<<endl;

#ifdef HOLE
	  if(v.size()<12) d.SF=d.sf, d.G=d.g, d.G2=d.g2, d.GR=d.gr;
	  else d.SF=v[10], d.G=v[11], d.G2=d.G*d.G, d.GR=(1-d.G)/(1+d.G);

	  if(v.size()>=10){
	    float xH=v[7], hS=v[8], hA=v[9];
	    d.hr=OMR*xH; d.hr2=d.hr*d.hr; d.hs=1/(hS*(1-d.G)), d.ha=1/hA;
	    if(xH>0) cerr<<"With hole ice: xH="<<xH<<" sca="<<hS<<" ("<<d.SF<<","<<d.G<<") abs="<<hA<<endl;
	  }
	  else d.hr=0, d.hr2=0, d.hs=0, d.ha=0;
#endif

#ifdef ANIZ
	  if(v.size()>=7){
	    const float cv=FPI/180, thx=v[4];
	    d.azx=cos(cv*thx), d.azy=sin(cv*thx);
	    d.k1=exp(v[5]); d.k2=exp(v[6]); d.kz=1/(d.k1*d.k2);
	    d.k=isfinite(d.kz) && ( d.k1!=1 || d.k2!=1 )?1:0;
	    if(d.k>0) cerr<<"Ice anisotropy is k("<<thx<<")="<<d.k1<<","<<d.k2<<","<<d.kz<<endl;
	    else cerr<<"Setting tilt direction to "<<thx<<endl;
	  }
	  else d.k1=1, d.k2=1, d.kz=1, d.k=0, d.azx=1, d.azy=0;
#endif
	}
	else{ cerr<<"File cfg.txt did not contain valid data"<<endl; exit(1); }
	inFile.close();
      }
      else{ cerr<<"Could not open file cfg.txt"<<endl; exit(1); }
    }

#ifdef ASENS
    {
      d.mas=1, d.s[0]=1;
      for(int i=1; i<ANUM; i++) d.s[i]=0;

      ifstream inFile((dir+"as.dat").c_str(), ifstream::in);
      if(!inFile.fail()){
	int n=0;
	float aux;
	if(inFile >> aux) d.mas=aux, n++;
	for(int i=0; i<ANUM; i++) if(inFile >> aux) d.s[i]=aux, n++;
	if(n>0) cerr<<"Loaded "<<n<<" angsens coefficients"<<endl;
	else{ cerr<<"File as.dat did not contain valid data"<<endl; exit(1); }
	inFile.close();
      }
      else{ cerr<<"Could not open file as.dat"<<endl; exit(1); }
      d.eff*=d.mas;
    }
#endif

#ifdef TILT
    {
      d.lnum=0; d.l0=0, d.r0=0;
      const float cv=FPI/180, thx=225;
      d.lnx=cos(cv*thx), d.lny=sin(cv*thx);

      ifstream inFile((dir+"tilt.par").c_str(), ifstream::in);
      if(!inFile.fail()){
	int str;
	float aux;
	vector<float> lr;
	while(inFile >> str >> aux){ if(aux==0) d.l0=str; lr.push_back(aux); }
	inFile.close();

	int size=lr.size();
	if(size>LMAX){ cerr << "File tilt.par defines too many dust maps" << endl; exit(1); }
	for(int i=1; i<size; i++) if(lr[i]<lr[i-1]) { cerr << "Tilt map does not use increasing range order" << endl; exit(1); };
	for(int i=0; i<size; i++) d.lr[i]=lr[i];

	ifstream inFile((dir+"tilt.dat").c_str(), ifstream::in);
	if(!inFile.fail()){
	  d.lnum=size;
	  float depth;
	  vector<float> pts(d.lnum), ds;
	  vector< vector<float> > lp(d.lnum);
	  while(inFile >> depth){
	    int i=0;
	    while(inFile >> pts[i++]) if(i>=d.lnum) break;
	    if(i!=d.lnum) break;
	    ds.push_back(depth);
	    for(i=0; i<d.lnum; i++) lp[i].push_back(pts[i]);
	  }
	  inFile.close();

	  int size=ds.size();
	  if(size>LYRS){ cerr << "File tilt.dat defines too many map points" << endl; exit(1); }
	  for(int i=1; i<size; i++) if(ds[i]<ds[i-1]) { cerr << "Tilt map does not use increasing depth order" << endl; exit(1); };
	  for(int i=0; i<d.lnum; i++) for(int j=0; j<size; j++) d.lp[i][j]=lp[i][size-1-j];
	  d.lpts=size;

	  if(size<2) d.lnum=0;
	  else{
	    float lmin=ds[0], lmax=ds[size-1];
	    d.lmin=zoff-lmax; d.lrdz=(d.lpts-1)/(lmax-lmin);
	  }
	}

	if(d.lnum>0) cerr<<"Loaded "<<d.lnum<<"x"<<d.lpts<<" dust layer points"<<endl;
      }
    }
#endif

    { // initialize random numbers
      int size;
      vector<unsigned int> rx;

      ifstream inFile((dir+"rnd.txt").c_str(), ifstream::in);
      if(!inFile.fail()){
	string in;
	while(getline(inFile, in)){
	  stringstream str(in);
	  unsigned int a;
	  if(str>>a) rx.push_back(a);
	}
	if(rx.size()<1){ cerr<<"File rnd.txt did not contain valid data"<<endl; exit(1); }
	inFile.close();
      }
      else{ cerr<<"Could not open file rnd.txt"<<endl; exit(1); }

      size=rx.size();
      if(size>MAXRND){
	cerr<<"Error: too many random multipliers ("<<size<<"), truncating to "<<MAXRND<<endl;
	size=MAXRND;
      }

      cerr<<"Loaded "<<size<<" random multipliers"<<endl;

#ifdef RAND
      timeval tv; gettimeofday(&tv, NULL);
      sv=1000000*(unsigned long long)tv.tv_sec+tv.tv_usec;
#endif

      d.rsize=size;
      for(int i=0; i<size; i++) z.rm[i]=rx[i];
    }

    { // initialize ice parameters
      int size;                 // size of kurt table
      float dh, hdh, rdh, hmin; // step, step/2, 1/step, and min depth

      vector<float> wx, wy, wz;

      {
	char * WFLA=getenv("WFLA");
	float wfla=WFLA==NULL?0:atof(WFLA);
	if(wfla>0){
	  wx.push_back(0); wy.push_back(wfla-XXX);
	  wx.push_back(1); wy.push_back(wfla+XXX);
          cerr<<"Using single wavelength="<<wfla<<" [nm]"<<endl;
	}
	else{
	  ifstream inFile((dir+"wv.dat").c_str(), ifstream::in);
	  if(!inFile.fail()){
	    int num=0;
	    bool flag=true;
	    float xa, ya, xo=0, yo=0;
	    while(inFile>>xa>>ya){
	      if(( xa<0 || 1<xa ) || num==0 && xa!=0 || num>0 && ( xa<=xo || ya<=yo )){ flag=false; break; }
	      wx.push_back(xa); wy.push_back(ya);
	      xo=xa; yo=ya; num++;
	    }
	    if(xo!=1 || wx.size()<2) flag=false;
	    inFile.close();
	    if(flag){ cerr<<"Loaded "<<wx.size()<<" wavelenth points"<<endl; }
	    else{ cerr<<"File wv.dat did not contain valid data"<<endl; exit(1); }
	  }
	  else{ cerr<<"Could not open file wv.dat"<<endl; exit(1); }
	}
      }

      {
	vector<float> qx, qy;

	ifstream inFile((dir+"wv.rde").c_str(), ifstream::in);
	if(!inFile.fail()){
	  int num=0;
	  bool flag=true;
	  float xa, ya, yo=0;
	  while(inFile>>ya>>xa){
	    if(xa<0 || num>0 && ya<=yo){ flag=false; break; }
	    qx.push_back(xa); qy.push_back(ya);
	    yo=ya; num++;
	  }
	  if(qx.size()<2) flag=false;
	  inFile.close();
	  if(flag){ cerr<<"Loaded "<<qx.size()<<" RDE coefficients"<<endl; }
	  else{ cerr<<"File wv.rde did not contain valid data"<<endl; exit(1); }
	  rdef=flag;
	}

	if(rdef){
	  int k=0, n=qy.size();
	  for(vector<float>::iterator i=wy.begin(); i!=wy.end(); i++){
	    float w=*i, r;
	    for(; k<n; k++) if(qy[k]>w) break;
	    if(k==0) r=qx[0];
	    else if(k==n) r=qx[n-1];
	    else{
	      r=((w-qy[k-1])*qx[k]+(qy[k]-w)*qx[k-1])/(qy[k]-qy[k-1]);
	      k--;
	    }
	    wz.push_back(r);
	  }

	  float eff=0, tmp=0;
	  wx[0]=0;

	  for(int i=1; i<wx.size(); i++){
	    eff+=(wx[i]-tmp)*max(1.f ,(wz[i]+wz[i-1])/2);
	    tmp=wx[i]; wx[i]=eff;
	  }

	  for(vector<float>::iterator i=wx.begin(); i!=wx.end(); i++) *i/=eff;
	  d.eff*=eff;
	}
      }

      float wv0=400;
      float A, B, D, E, a, k;
      float Ae, Be, De, Ee, ae, ke;
      vector<float> dp, be, ba, td;

      {
	bool flag=true, fail=false;
	ifstream inFile((dir+"icemodel.par").c_str(), ifstream::in);
	if(flag=!inFile.fail()){
	  if(flag) flag=static_cast<bool>(inFile >> a >> ae);
	  if(flag) flag=static_cast<bool>(inFile >> k >> ke);
	  if(flag) flag=static_cast<bool>(inFile >> A >> Ae);
	  if(flag) flag=static_cast<bool>(inFile >> B >> Be); fail=!flag;
	  if(flag) flag=static_cast<bool>(inFile >> D >> De); if(!flag) D=pow(wv0, k);
	  if(flag) flag=static_cast<bool>(inFile >> E >> Ee); if(!flag) E=0;
	  if(fail) cerr << "File icemodel.par found, but is corrupt" << endl;
	  inFile.close(); if(fail) exit(1);
	}
	else{ cerr << "File icemodel.par was not found" << endl; exit(1); }
      }

      {
	ifstream inFile((dir+"icemodel.dat").c_str(), ifstream::in);
	if(!inFile.fail()){
	  size=0;
	  float dpa, bea, baa, tda;
	  while(inFile >> dpa >> bea >> baa >> tda){
	    dp.push_back(dpa);
	    be.push_back(bea);
	    ba.push_back(baa);
	    td.push_back(tda);
	    size++;
	  }
	  inFile.close();
	  if(size<2){ cerr << "File icemodel.dat found, but is corrupt" << endl; exit(1); }
	}
	else{ cerr << "File icemodel.dat was not found" << endl; exit(1); }
      }

      dh=dp[1]-dp[0];
      if(dh<=0){ cerr << "Ice table does not use increasing depth spacing" << endl; exit(1); }

      for(int i=0; i<size; i++) if(i>0) if(fabsf(dp[i]-dp[i-1]-dh)>dh*XXX){
	cerr << "Ice table does not use uniform depth spacing" << endl; exit(1);
      }
      cerr<<"Loaded "<<size<<" ice layers"<<endl;

      if(size>MAXLYS){
	cerr<<"Error: too many layers ("<<size<<"), truncating to "<<MAXLYS<<endl;
	size=MAXLYS;
      }

      hdh=dh/2; rdh=1/dh; hmin=zoff-dp[size-1];
      {
	d.size=size;
	d.dh=dh;
	d.hdh=hdh;
	d.rdh=rdh;
	d.hmin=hmin;
      }

      for(int n=0; n<WNUM; n++){
	float p=(n+0.5f)/WNUM;
	int m=0; while(wx[++m]<p);
	float wva=(wy[m-1]*(wx[m]-p)+wy[m]*(p-wx[m-1]))/(wx[m]-wx[m-1]);
	if(rdef){
	  float rde=(wz[m-1]*(wx[m]-p)+wz[m]*(p-wx[m-1]))/(wx[m]-wx[m-1]);
	  q.rde.insert(make_pair(wva, rde));
	}

	float l_a=pow(wva/wv0, -a);
	float l_k=pow(wva, -k);
	float ABl=A*exp(-B/wva);

	ices & w = z.w[n];

	for(int i=0; i<size; i++){
	  int j=size-1-i;
	  float sca=be[j]*l_a/(1-d.g), abs=(D*ba[j]+E)*l_k+ABl*(1+0.01*td[j]);
	  if(sca>0 && abs>0) w.z[i].sca=sca, w.z[i].abs=abs;
	  else{ cerr << "Invalid value of ice parameter, cannot proceed" << endl; exit(1); }
	}

	float wv=wva*1.e-3;
	float wv2=wv*wv;
	float wv3=wv*wv2;
	float wv4=wv*wv3;
	float np=1.55749-1.57988*wv+3.99993*wv2-4.68271*wv3+2.09354*wv4;
	float ng=np*(1+0.227106-0.954648*wv+1.42568*wv2-0.711832*wv3);
	float c=0.299792458; d.ocv=1/c; w.wvl=wva; w.ocm=ng/c;
	w.coschr=1/np; w.sinchr=sqrt(1-w.coschr*w.coschr);
      }
      d.fla=-1;

      {
	char * FLDR=getenv("FLDR");
	d.fldr=FLDR==NULL?-1:atof(FLDR);
	if(d.fldr>=0){
	  float fold=int(d.fldr/360);
	  float dir=d.fldr-360*fold++;
	  cerr<<"Flasher LEDs are in a "<<fold<<"-fold pattern with LED #0 at "<<dir<<" degrees"<<endl;
	}
      }
    }

    {
      ifstream inFile((dir+"geo-f2k").c_str(), ifstream::in);
      if(!inFile.fail()){
	OM om;
	string mbid;
	unsigned long long omid;
	while(inFile>>mbid>>hex>>omid>>dec>>om.r[0]>>om.r[1]>>om.r[2]>>om.str>>om.dom){
	  om.r[2]+=zoff;
	  i3oms.push_back(om);
	}
	inFile.close();
      }
    }
    {
      ifstream inFile((dir+"eff-f2k").c_str(), ifstream::in);
      if(!inFile.fail()){
	ikey om;
	float eff;
	while(inFile>>om.str>>om.dom>>eff) if(om.isinice()) rdes.insert(make_pair(om, eff));
	inFile.close();
      }
    }
    {
      ifstream inFile((dir+"hvs-f2k").c_str(), ifstream::in);
      if(!inFile.fail()){
	ikey om;
	float hv;
	while(inFile>>om.str>>om.dom>>hv) if(om.isinice()) hvs.insert(make_pair(om, hv));
	inFile.close();
      }
    }

    { // initialize geometry
      vector<DOM> oms;
      vector<name> names;
      int nhqe=0;

      sort(i3oms.begin(), i3oms.end());
      for(vector<OM>::const_iterator i=i3oms.begin(); i!=i3oms.end(); ++i) if(i->isinice()){
	oms.push_back(*i);
	ikey om(*i);

	float r, h;
	{
	  map<ikey, float>::iterator j=rdes.find(om);
	  if(j!=rdes.end()){
	    nhqe++;
	    r=j->second;
	    if(r>rmax) rmax=r;
	  }
	  else r=1;
	}
	if(hvs.empty()) h=1200;
	else{
	  map<ikey, float>::iterator j=hvs.find(om);
	  h=j==hvs.end()?0:j->second;
	}

	names.push_back(name(om, r, h));
      }
      if(nhqe>0) cerr<<"Loaded "<<nhqe<<" high-QE DOMs"<<endl;
      if(!rdef) d.eff*=rmax;

      int gsize = oms.size();
      if(gsize>MAXGEO){
	cerr<<"Error: too many OMs ("<<gsize<<"), truncating to "<<MAXGEO<<endl;
	gsize=MAXGEO;
      }

      for(int n=0; n<gsize; n++){ q.oms[n]=oms[n]; q.names[n]=names[n]; }

      d.gsize=gsize;
    }

    map<unsigned char, short> num;
    {
      map<unsigned char, float> l;
      map<unsigned char, line> sc;
      for(int n=0; n<d.gsize; n++){
	unsigned char str=sname(n);
	line & s = sc[str];
	DOM & om = q.oms[n];
	if(num.find(str)==num.end()){
	  l[str]=om.r[2];
	  s.h=om.r[2];
	  s.n=n;
	}
	else{
	  if(l[str]>om.r[2]) l[str]=om.r[2];
	  if(s.h<om.r[2]) s.h=om.r[2];
	  if(s.n>n) s.n=n;
	}
	num[str]++; s.x+=om.r[0], s.y+=om.r[1];
      }
      if(sc.size()>NSTR){ cerr<<"Number of strings exceeds capacity of "<<NSTR<<endl; exit(1); }

      for(map<unsigned char, short>::iterator i=num.begin(); i!=num.end(); ++i){
	unsigned char str=i->first, n=i->second;
	line & s = sc[str];
	float d=s.h-l[str];
	if(n>1 && d<=0){ cerr<<"Cannot estimate the spacing along string "<<(int)str<<endl; exit(1); }
	s.x/=n, s.y/=n, s.r=0; s.d=n>1?(n-1)/d:0; s.dl=0; s.dh=0;
      }

      for(int n=0; n<d.gsize; n++){
	unsigned char str=sname(n);
	line & s = sc[str];
	DOM & om = q.oms[n];
#ifdef TILT
	if(str==d.l0) d.r0=d.lnx*s.x+d.lny*s.y;
#endif

	float dx=s.x-om.r[0], dy=s.y-om.r[1];
	float dr=dx*dx+dy*dy; if(dr>s.r) s.r=dr;

	if(s.d>0){
	  float dz=om.r[2]-(s.h+(s.n-n)/s.d);
	  if(s.dl>dz) s.dl=dz; if(s.dh<dz) s.dh=dz;
	}
      }

      d.rx=0;
      int n=0;
      for(map<unsigned char, short>::iterator i=num.begin(); i!=num.end(); ++i, n++){
	unsigned char str=i->first;
	line & s = sc[str];
	s.max=i->second-1;
	i->second=n;
	s.r=d.R+sqrt(s.r);
#ifdef HOLE
	if(d.hr>s.r) s.r=d.hr;
#endif
	if(d.rx<s.r) d.rx=s.r;
	s.dl-=d.R, s.dh+=d.R;
	d.sc[n]=s;
      }
    }

    float sin12=0;
    {
#ifdef ROMB
      const float cv=FPI/180;
      float bv[2][2];
      bv[0][0]=cos(cv*DIR1);
      bv[0][1]=sin(cv*DIR1);
      bv[1][0]=cos(cv*DIR2);
      bv[1][1]=sin(cv*DIR2);

      float det=bv[0][0]*bv[1][1]-bv[0][1]*bv[1][0];
      d.cb[0][0]=bv[1][1]/det;
      d.cb[0][1]=-bv[0][1]/det;
      d.cb[1][0]=-bv[1][0]/det;
      d.cb[1][1]=bv[0][0]/det;

      for(int i=0; i<2; i++) sin12+=bv[0][i]*bv[1][i];
#endif
      sin12=sqrt(1-sin12*sin12);
      d.rx/=sin12;
    }

    map<unsigned char, int> cells[CX][CY];
    {
      float cl[2]={0,0}, ch[2]={0,0}, crst[2];

      int n=0;
      for(map<unsigned char, short>::iterator i=num.begin(); i!=num.end(); ++i, n++){
	line & s = d.sc[i->second];
	for(int m=0; m<2; m++){
	  if(n==0 || ctr(s, m)<cl[m]) cl[m]=ctr(s, m);
	  if(n==0 || ctr(s, m)>ch[m]) ch[m]=ctr(s, m);
	}
      }

      d.cn[0]=CX;
      d.cn[1]=CY;

      for(int m=0; m<2; m++){
	float diff=ch[m]-cl[m];
#ifdef ROMB
	d.cn[m]=min(d.cn[m], 1+2*(int)lroundf(diff/125));
#endif
	if(d.cn[m]<=1){
	  ch[m]=cl[m]=(cl[m]+ch[m])/2;
	  crst[m]=1/(d.rx*(2+XXX)+diff);
	}
	else{
	  float s=d.R*(d.cn[m]-1);
	  if(diff<2*s){
	    cerr<<"Warning: tight string packing in direction "<<(m<1?"x":"y")<<endl;
	    float ave=(cl[m]+ch[m])/2;
	    cl[m]=ave-s; ch[m]=ave+s; diff=2*s;
	  }
	  crst[m]=(d.cn[m]-1)/diff;
	}
      }

      bool flag=true;
      for(map<unsigned char, short>::iterator i=num.begin(); i!=num.end(); ++i){
	line & s = d.sc[i->second];
	int n[2];
	for(int m=0; m<2; m++){
	  n[m]=lroundf((ctr(s, m)-cl[m])*crst[m]);
	  if(n[m]<0 && n[m]>=d.cn[m]){ cerr<<"Error in cell initialization"<<endl; exit(1); }

	  float d1=fabsf(ctr(s, m)-(cl[m]+(n[m]-0.5f)/crst[m]));
	  float d2=fabsf(ctr(s, m)-(cl[m]+(n[m]+0.5f)/crst[m]));
	  float d=min(d1, d2)*sin12-s.r;
	  if(d<0){ flag=false; cerr<<"Warning: string "<<(int)i->first<<" too close to cell boundary"<<endl; }
	}

	cells[n[0]][n[1]][i->first]++;
      }
      if(flag) d.rx=0;

      for(int m=0; m<2; m++){ d.cl[m]=cl[m]; d.crst[m]=crst[m]; }
    }

    {
      unsigned int pos=0;
      for(int i=0; i<d.cn[0]; i++) for(int j=0; j<d.cn[1]; j++){
	map<unsigned char, int> & c = cells[i][j];

	if(c.size()>0){
	  d.is[i][j]=pos;
	  for(map<unsigned char, int>::const_iterator n=c.begin(); n!=c.end(); ++n){
	    if(pos==NSTR){ cerr<<"Number of string cells exceeds capacity of "<<NSTR<<endl; exit(1); }
	    d.ls[pos++]=num[n->first];
	  }
	  d.ls[pos-1]|=0x80;
	}
	else d.is[i][j]=0x80;
      }
    }

    cerr<<"Loaded "<<d.gsize<<" DOMs ("<<d.cn[0]<<"x"<<d.cn[1]<<")"<<endl<<endl;
  }
} m;
