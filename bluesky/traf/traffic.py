import numpy as np
from math import *

from ..tools.aero import fpm, kts, ft, nm, g0,  tas2eas, tas2mach, tas2cas, mach2cas,  \
     mach2tas, cas2tas, Rearth

from ..tools.aero_np import vatmos, vcas2tas, vtas2cas,  vtas2mach, cas2mach, \
    vmach2tas, qdrdist
from ..tools.misc import degto180
from ..tools.datalog import Datalog

from route import Route
from params import Trails
from adsbmodel import ADSBModel
from asas import Dbconf
from .. import settings

try:
    if settings.performance_model == 'bluesky':
        from perf import Perf

    elif settings.performance_model == 'bada':
        from perfbada import PerfBADA as Perf

except ImportError as e:
    print e.args[0]
    print 'Falling back to BlueSky performance model'
    from perf import Perf


class Traffic:
    """
    Traffic class definition    : Traffic data

    Methods:
        Traffic()            :  constructor

        create(acid,actype,aclat,aclon,achdg,acalt,acspd) : create aircraft
        delete(acid)         : delete an aircraft from traffic data
        update(sim)          : do a numerical integration step
        trafperf ()          : calculate aircraft performance parameters

    Members: see create

    Created by  : Jacco M. Hoekstra
    """

    def __init__(self, navdb):

        self.reset(navdb)
        return

    def reset(self, navdb):
        self.dts = []
        self.ntraf = 0

        #  model-specific parameters.
        # Default: BlueSky internal performance model.
        # Insert your BADA files to the folder "BlueSky/data/coefficients/BADA"
        # for working with EUROCONTROL`s Base of Aircraft Data revision 3.12

        self.perf = Perf(self)

        self.dts = []

        self.ntraf = 0

        # Create datalog instance
        self.log = Datalog()

        # Traffic list & arrays definition

        # !!!IMPORTANT NOTE!!!
        # Any variables added here should also be added in the Traffic
        # methods self.create() (append) and self.delete() (delete)
        # which can be found directly below __init__

        # Traffic basic flight data

        # Traffic basic flight data
        self.id     = []  # identifier (string)
        self.type   = []  # aircaft type (string)
        self.lat    = np.array([])  # latitude [deg]
        self.lon    = np.array([])  # longitude [deg]
        self.trk    = np.array([])  # track angle [deg]
        self.tas    = np.array([])  # true airspeed [m/s]
        self.gs     = np.array([])  # ground speed [m/s]
        self.cas    = np.array([])  # callibrated airspeed [m/s]
        self.M      = np.array([])  # mach number
        self.alt    = np.array([])  # altitude [m]
        self.fll    = np.array([])  # flight level [ft/100]
        self.vs     = np.array([])  # vertical speed [m/s]
        self.p      = np.array([])  # atmospheric air pressure [N/m2]
        self.rho    = np.array([])  # atmospheric air density [kg/m3]
        self.Temp   = np.array([])  # atmospheric air temperature [K]
        self.dtemp  = np.array([])  # delta t for non-ISA conditions

        # Traffic performance data
        self.avsdef = np.array([])  # [m/s]default vertical speed of autopilot
        self.aphi   = np.array([])  # [rad] bank angle setting of autopilot
        self.ax     = np.array([])  # [m/s2] absolute value of longitudinal accelleration
        self.bank   = np.array([])  # nominal bank angle, [radian]
        self.bphase = np.array([])  # standard bank angles per phase
        self.hdgsel = np.array([])  # determines whether aircraft is turning

        # Crossover altitude
        self.abco   = np.array([])
        self.belco  = np.array([])

        # Traffic autopilot settings
        self.ahdg   = []  # selected heading [deg]
        self.aspd   = []  # selected spd(eas) [m/s]
        self.aptas  = []  # just for initializing
        self.ama    = []  # selected spd above crossover altitude (Mach) [-]
        self.aalt   = []  # selected alt[m]
        self.afll   = []  # selected fl [ft/100]
        self.avs    = []  # selected vertical speed [m/s]

        # limit settings
        self.lspd   = []  # limit speed
        self.lalt   = []  # limit altitude
        self.lvs    = []  # limit vertical speed due to thrust limitation

        # Traffic navigation information
        self.orig   = []  # Four letter code of origin airport
        self.dest   = []  # Four letter code of destination airport

        # LNAV route navigation
        self.swlnav = np.array([])  # Lateral (HDG) based on nav?
        self.swvnav = np.array([])  # Vertical/longitudinal (ALT+SPD) based on nav info

        self.actwplat  = np.array([])  # Active WP latitude
        self.actwplon  = np.array([])  # Active WP longitude
        self.actwpalt  = np.array([])  # Active WP altitude to arrive at
        self.actwpspd  = np.array([])  # Active WP speed
        self.actwpturn = np.array([])  # Distance when to turn to next waypoint

        # VNAV cruise level
        self.crzalt = np.array([])    # Cruise altitude[m]

        # Route info
        self.route = []

        # ASAS info per aircraft:
        self.iconf      = []            # index in 'conflicting' aircraft database
        self.asasactive = np.array([])  # whether the autopilot follows ASAS or not
        self.asashdg    = np.array([])  # heading provided by the ASAS [deg]
        self.asasspd    = np.array([])  # speed provided by the ASAS (eas) [m/s]
        self.asasalt    = np.array([])  # speed alt by the ASAS [m]
        self.asasvsp    = np.array([])  # speed vspeed by the ASAS [m/s]

        self.desalt     = np.array([])  # desired altitude [m]
        self.deshdg     = np.array([])  # desired heading
        self.desvs      = np.array([])  # desired vertical speed [m/s]
        self.desspd     = np.array([])  # desired speed [m/s]

        # Display information on label
        self.label      = []  # Text and bitmap of traffic label
        self.trailcol   = []  # Trail color: default 'Blue'

        # Transmitted data to other aircraft due to truncated effect
        self.adsbtime   = np.array([])
        self.adsblat    = np.array([])
        self.adsblon    = np.array([])
        self.adsbalt    = np.array([])
        self.adsbtrk    = np.array([])
        self.adsbtas    = np.array([])
        self.adsbgs     = np.array([])
        self.adsbvs     = np.array([])

        self.inconflict = np.array([], dtype=bool)

        #-----------------------------------------------------------------------------
        # Not per aircraft data

        # Scheduling of FMS and ASAS
        self.t0fms = -999.  # last time fms was called
        self.dtfms = 1.01  # interval for fms

        self.t0asas = -999.  # last time ASAS was called
        self.dtasas = 1.00  # interval for ASAS

        # Flight performance scheduling
        self.perfdt = 0.1           # [s] update interval of performance limits
        self.perft0 = -self.perfdt  # [s] last time checked (in terms of simt)
        self.warned2 = False        # Flag: Did we warn for default engine parameters yet?

        # ADS-B transmission-receiver model
        self.adsb = ADSBModel(self)

        # ASAS objects: Conflict Database
        self.dbconf = Dbconf(self, 300., 5. * nm, 1000. * ft)  # hard coded values to be replaced

        # Import navigation data base
        self.navdb  = navdb

        # Traffic area: delete traffic when it leaves this area (so not when outside)
        self.swarea    = False
        self.arealat0  = 0.0  # [deg] lower latitude defining area
        self.arealat1  = 0.0  # [deg] upper latitude defining area
        self.arealat0  = 0.0  # [deg] lower longitude defining area
        self.arealat1  = 0.0  # [deg] upper longitude defining area
        self.areafloor = -999999.0  # [m] Delete when descending through this h
        self.areadt    = 5.0  # [s] frequency of area check (simtime)
        self.areat0    = -100.  # last time checked

        self.inside = []
        self.fir_circle_point = (0.0, 0.0)
        self.fir_circle_radius = 1.0

        # Taxi switch
        self.swtaxi = False  # Default OFF: delete traffic below 1500 ft

        # Research Area ("Square" for Square, "Circle" for Circle area)
        self.area = ""

        # Bread crumbs for trails
        self.lastlat  = []
        self.lastlon  = []
        self.lasttim  = []
        self.trails   = Trails()
        self.swtrails = False  # Default switched off

        # ADS-B Coverage area
        self.swAdsbCoverage = False

        # Noise (turbulence, ADBS-transmission noise, ADSB-truncated effect)
        self.setNoise(False)

        self.eps = np.array([])

        return

    def create_ac(self, acid, actype, aclat, aclon, achdg, acalt, acspd):
        """Create an aircraft"""
        # Check if not already exist
        if self.id.count(acid.upper()) > 0:
            return False  # already exists do nothing

        # Increase number of aircraft
        self.ntraf = self.ntraf + 1

        # Process input
        self.id.append(acid.upper())
        self.type.append(actype)
        self.lat   = np.append(self.lat, aclat)
        self.lon   = np.append(self.lon, aclon)
        self.trk   = np.append(self.trk, achdg)  # TBD: add conversion hdg => trk
        self.alt   = np.append(self.alt, acalt)
        self.fll   = np.append(self.fll, (acalt)/100)
        self.vs    = np.append(self.vs, 0.)
        c_temp, c_rho, c_p = vatmos(acalt)
        self.p     = np.append(self.p, c_p)
        self.rho   = np.append(self.rho, c_rho)
        self.Temp  = np.append(self.Temp, c_temp)
        self.dtemp = np.append(self.dtemp, 0)  # at the moment just ISA conditions
        self.tas   = np.append(self.tas, acspd)
        self.gs    = np.append(self.gs, acspd)
        self.cas   = np.append(self.cas, tas2cas(acspd, acalt))
        self.M     = np.append(self.M, tas2mach(acspd, acalt))

        # AC is initialized with neutral max bank angle
        self.bank = np.append(self.bank, 25.)
        if self.ntraf < 2:
            self.bphase = np.deg2rad(np.array([15, 35, 35, 35, 15, 45]))
        self.hdgsel = np.append(self.hdgsel, False)

        #------------------------------Performance data--------------------------------
        # Type specific data
        #(temporarily default values)
        self.avsdef = np.append(self.avsdef, 1500. * fpm)  # default vertical speed of autopilot
        self.aphi   = np.append(self.aphi, radians(25.))  # bank angle setting of autopilot
        self.ax     = np.append(self.ax, kts)  # absolute value of longitudinal accelleration

        # Crossover altitude
        self.abco   = np.append(self.abco, 0)
        self.belco  = np.append(self.belco, 1)

        # performance data
        self.perf.create(actype)

        # Traffic autopilot settings: hdg[deg], spd (CAS,m/s), alt[m], vspd[m/s]
        self.ahdg = np.append(self.ahdg, achdg)  # selected heading [deg]
        self.aspd = np.append(self.aspd, tas2eas(acspd, acalt))  # selected spd(eas) [m/s]
        self.aptas = np.append(self.aptas, vcas2tas(self.aspd, self.alt)) # [m/s]
        self.ama  = np.append(self.ama, 0.) # selected spd above crossover (Mach) [-]
        self.aalt = np.append(self.aalt, acalt)  # selected alt[m]
        self.afll = np.append(self.afll, (acalt/100)) # selected fl[ft/100]
        self.avs = np.append(self.avs, 0.)  # selected vertical speed [m/s]
        
        # limit settings: initialize with 0
        self.lspd = np.append(self.lspd, 0.0)
        self.lalt = np.append(self.lalt, 0.0)
        self.lvs = np.append(self.lvs, 0.0)

        # Traffic navigation information
        self.dest.append("")
        self.orig.append("")

        # LNAV route navigation
        self.swlnav = np.append(self.swlnav, False)  # Lateral (HDG) based on nav
        self.swvnav = np.append(self.swvnav, False)  # Vertical/longitudinal (ALT+SPD) based on nav info

        self.actwplat  = np.append(self.actwplat, 89.99)  # Active WP latitude
        self.actwplon  = np.append(self.actwplon, 0.0)   # Active WP longitude
        self.actwpalt  = np.append(self.actwpalt, 0.0)   # Active WP altitude
        self.actwpspd  = np.append(self.actwpspd, -999.)   # Active WP speed
        self.actwpturn = np.append(self.actwpturn, 1.0)   # Distance to active waypoint where to turn

        # VNAV cruise level
        self.crzalt = np.append(self.crzalt,-999.) # Cruise altitude[m] <0=None

        # Route info
        self.route.append(Route(self.navdb))  # create empty route connected with nav databse

        eas = tas2eas(acspd, acalt)
        # ASAS info: no conflict => -1
        self.iconf.append(-1)  # index in 'conflicting' aircraft database
        self.asasactive = np.append(self.asasactive, False)
        self.asashdg = np.append(self.asashdg, achdg)
        self.asasspd = np.append(self.asasspd, eas)
        self.asasalt = np.append(self.asasalt, acalt)
        self.asasvsp = np.append(self.asasvsp, 0.)

        self.desalt  = np.append(self.desalt, acalt)
        self.desvs   = np.append(self.desvs, 0.0)
        self.desspd  = np.append(self.desspd, eas)
        self.deshdg  = np.append(self.deshdg, achdg)

        # Area variable set to False to avoid deletion upon creation outside
        self.inside.append(False)

        # Display information on label
        self.label.append(['', '', '', 0])

        # Bread crumbs for trails
        self.trailcol.append(self.trails.defcolor)
        self.lastlat = np.append(self.lastlat, aclat)
        self.lastlon = np.append(self.lastlon, aclon)
        self.lasttim = np.append(self.lasttim, 0.0)

        # ADS-B Coverage area
        self.swAdsbCoverage = False
        
        # Transmitted data to other aircraft due to truncated effect
        self.adsbtime=np.append(self.adsbtime,np.random.rand(self.trunctime))
        self.adsblat=np.append(self.adsblat,aclat)
        self.adsblon=np.append(self.adsblon,aclon)
        self.adsbalt=np.append(self.adsbalt,acalt)
        self.adsbtrk=np.append(self.adsbtrk,achdg)
        self.adsbtas=np.append(self.adsbtas,acspd)
        self.adsbgs=np.append(self.adsbgs,acspd)
        self.adsbvs=np.append(self.adsbvs,0.)
        
        self.inconflict=np.append(self.inconflict,False)        
        
        self.eps = np.append(self.eps, 0.01)

        return True

    def delete(self, acid):
        """Delete an aircraft"""

        try:  # prevent error due to not found
            idx = self.id.index(acid)
        except:
            return False

        del self.id[idx]
        del self.type[idx]

        # Traffic basic data
        self.lat    = np.delete(self.lat, idx)
        self.lon    = np.delete(self.lon, idx)
        self.trk    = np.delete(self.trk, idx)
        self.alt    = np.delete(self.alt, idx)
        self.fll    = np.delete(self.fll, idx)
        self.vs     = np.delete(self.vs, idx)
        self.tas    = np.delete(self.tas, idx)
        self.gs     = np.delete(self.gs, idx)
        self.cas    = np.delete(self.cas, idx)
        self.M      = np.delete(self.M, idx)

        self.p      = np.delete(self.p, idx)
        self.rho    = np.delete(self.rho, idx)
        self.Temp   = np.delete(self.Temp, idx)
        self.dtemp  = np.delete(self.dtemp, idx)
        self.hdgsel = np.delete(self.hdgsel, idx)
        self.bank   = np.delete(self.bank, idx)

        # Crossover altitude
        self.abco   = np.delete(self.abco, idx)
        self.belco  = np.delete(self.belco, idx)

        # Type specific data (temporarily default values)
        self.avsdef = np.delete(self.avsdef, idx)
        self.aphi   = np.delete(self.aphi, idx)
        self.ax     = np.delete(self.ax, idx)

        # performance data
        self.perf.delete(idx)

        # Traffic autopilot settings: hdg[deg], spd (CAS,m/s), alt[m], vspd[m/s]
        self.ahdg   = np.delete(self.ahdg, idx)
        self.aspd   = np.delete(self.aspd, idx)
        self.ama    = np.delete(self.ama, idx)
        self.aptas  = np.delete(self.aptas, idx)
        self.aalt   = np.delete(self.aalt, idx)
        self.afll   = np.delete(self.afll, idx)
        self.avs    = np.delete(self.avs, idx)

        # limit settings
        self.lspd   = np.delete(self.lspd, idx)
        self.lalt   = np.delete(self.lalt, idx)
        self.lvs    = np.delete(self.lvs, idx)

        # Traffic navigation variables
        del self.dest[idx]
        del self.orig[idx]

        self.swlnav = np.delete(self.swlnav, idx)
        self.swvnav = np.delete(self.swvnav, idx)

        self.actwplat  = np.delete(self.actwplat,  idx)
        self.actwplon  = np.delete(self.actwplon,  idx)
        self.actwpalt  = np.delete(self.actwpalt,  idx)
        self.actwpspd  = np.delete(self.actwpspd,  idx)
        self.actwpturn = np.delete(self.actwpturn, idx)

        # VNAV cruise level
        self.crzalt    = np.delete(self.crzalt,    idx)

        # Route info
        del self.route[idx]

        # ASAS info
        del self.iconf[idx]
        self.asasactive = np.delete(self.asasactive, idx)
        self.asashdg    = np.delete(self.asashdg, idx)
        self.asasspd    = np.delete(self.asasspd, idx)
        self.asasalt    = np.delete(self.asasalt, idx)
        self.asasvsp    = np.delete(self.asasvsp, idx)

        self.desalt     = np.delete(self.desalt, idx)
        self.desvs      = np.delete(self.desvs, idx)
        self.desspd     = np.delete(self.desspd, idx)
        self.deshdg     = np.delete(self.deshdg, idx)

        # Metrics, area
        del self.inside[idx]

        # Traffic display data: label
        del self.label[idx]

        # Delete bread crumb data
        self.lastlat = np.delete(self.lastlat, idx)
        self.lastlon = np.delete(self.lastlon, idx)
        self.lasttim = np.delete(self.lasttim, idx)
        del self.trailcol[idx]

        # Transmitted data to other aircraft due to truncated effect
        self.adsbtime = np.delete(self.adsbtime, idx)
        self.adsblat  = np.delete(self.adsblat, idx)
        self.adsblon  = np.delete(self.adsblon, idx)
        self.adsbalt  = np.delete(self.adsbalt, idx)
        self.adsbtrk  = np.delete(self.adsbtrk, idx)
        self.adsbtas  = np.delete(self.adsbtas, idx)
        self.adsbgs   = np.delete(self.adsbgs, idx)
        self.adsbvs   = np.delete(self.adsbvs, idx)

        self.inconflict = np.delete(self.inconflict, idx)

        # Decrease number fo aircraft
        self.ntraf = self.ntraf - 1

        self.eps = np.delete(self.eps, idx)
        return True

    def deleteall(self):
        """Clear traffic buffer"""
        ndel = self.ntraf
        for i in range(ndel):
            self.delete(self.id[-1])
        self.ntraf = 0
        self.dbconf.reset()
        self.perf.reset()
        return

    def update(self, simt, simdt):
#        i = self.id2idx("TP233")
#        if i>=0:
#           print "LNAV TP233:",self.swlnav[i]
        # Update only necessary if there is traffic
        if self.ntraf == 0:
            return

        self.dts.append(simdt)

        #---------------- Atmosphere ----------------
        self.Temp, self.rho, self.p = vatmos(self.alt)

        #-------------- Performance limits autopilot settings --------------
        # Check difference with AP settings for trafperf and autopilot
        self.delalt = self.aalt - self.alt  # [m]
        
        # below crossover altitude: CAS=const, above crossover altitude: MA = const
        # aptas hast to be calculated before delspd
        self.aptas = vcas2tas(self.aspd, self.alt)*self.belco + vmach2tas(self.ama, self.alt)*self.abco  
        self.delspd = self.aptas - self.tas 
   

        ###############################################################################
        # Debugging: add 10000 random aircraft
        #            if simt>1.0 and self.ntraf<1000:
        #                for i in range(10000):
        #                   acid="KL"+str(i)
        #                   aclat = random.random()*180.-90.
        #                   aclon = random.random()*360.-180.
        #                   achdg = random.random()*360.
        #                   acalt = (random.random()*18000.+2000.)*0.3048
        #                   self.create(acid,'B747',aclat,aclon,achdg,acalt,350.)
        #
        #################################################################################
        
        #-------------------- ADSB update: --------------------

        self.adsbtime = self.adsbtime + simdt
        if self.ADSBtrunc:
            ADSB_update = np.where(self.adsbtime>self.trunctime)
        else:
            ADSB_update = range(self.ntraf)

        for i in ADSB_update:
            self.adsbtime[i] = self.adsbtime[i] - self.trunctime
            self.adsblat[i]  = self.lat[i]
            self.adsblon[i]  = self.lon[i]
            self.adsbalt[i]  = self.alt[i]
            self.adsbtrk[i]  = self.trk[i]
            self.adsbtas[i]  = self.tas[i]
            self.adsbgs[i]   = self.gs[i]
            self.adsbvs[i]   = self.vs[i]

        # New version ADSB Model
        self.adsb.update()        

        #------------------- ASAS update: ---------------------
        # Scheduling: when dt has passed or restart:
        if self.t0asas+self.dtasas<simt or simt<self.t0asas \
            and self.dbconf.swasas:
            self.t0asas = simt

            # Save old result
            iconf0 = np.array(self.iconf)

            # Call with traffic database and sim data
            self.dbconf.detect()
            self.dbconf.conflictlist(simt)
            self.dbconf.APorASAS()
            self.dbconf.resolve()

            # Reset label because of colour change
            chnged = np.where(iconf0!=np.array(self.iconf))[0]
            for i in chnged:
                self.label[i]=[" "," ", ""," "]


        #-----------------  FMS GUIDANCE & NAVIGATION  ------------------
        # Scheduling: when dt has passed or restart:
        if self.t0fms+self.dtfms<simt or simt<self.t0fms:
            self.t0fms = simt
            
            # FMS LNAV mode:

            qdr, dist = qdrdist(self.lat, self.lon, self.actwplat, self.actwplon) #[deg][nm]

            # Check whether shift based dist [nm] is required, set closer than WP turn distance

            iwpclose = np.where(self.swlnav*(dist < self.actwpturn))[0]
            
            # Shift for aircraft i where necessary

            for i in iwpclose:

            # Get next wp (lnavon = False if no more waypoints)

                lat, lon, alt, spd, xtoalt, toalt, lnavon =  \
                       self.route[i].getnextwp()  # note: xtoalt,toalt in [m]

            # End of route/no more waypoints: switch off LNAV

                if not lnavon:
                    self.swlnav[i] = False # Drop LNAV at end of route

            # In case of no LNAV, do not allow VNAV mode on it sown

                if not self.swlnav[i]:
                    self.swvnav[i] = False
                    
                self.actwplat[i] = lat
                self.actwplon[i] = lon

                # User entered altitude

                if alt >= 0.:
                    self.actwpalt[i] = alt
                    
                # VNAV calculated altitude is available and active
                
                if toalt  >= 0. and self.swvnav[i]:
                    # Descent VNAV mode (T/D logic)
                    
                    if self.alt[i]>toalt+10.*ft:       # Descent part is in this range of waypoints:

                        # Flat earth distance to next wp
                        dy = (lat-self.lat[i])
                        dx = (lon-self.lon[i])*cos(radians(lat))
                        dist2wp = 60.*nm*sqrt(dx*dx+dy*dy)
              
                        steepness = 3000.*ft/(10.*nm) # 1:3 rule of thumb for now
                        maxaltwp  = toalt + xtoalt*steepness    # max allowed altitude at next wp
                        self.actwpalt[i] = min(self.alt[i],maxaltwp) #To descend now or descend later?

                        if maxaltwp<self.alt[i]: # if descent is necessary with maximum steepness

                            self.aalt[i] = self.actwpalt[i] # dial in altitude of next waypoint as calculated

                            t2go         = max(0.1,dist2wp)/max(0.01,self.gs[i])
                            self.avs[i]  = (self.actwpalt[i] - self.alt[i])/t2go
                       
                           
                        else:
                            print "else 1"
                            pass # TBD

                    # Climb VNAV mode: climb as soon as possible (T/C logic)                        

                    elif self.swvnav[i] and self.alt[i]<toalt-10.*ft:

                        # Flat earth distance to next wp

                        dy = (lat-self.lat[i])
                        dx = (lon-self.lon[i])*cos(radians(lat))
                        dist2wp = 60.*nm*sqrt(dx*dx+dy*dy)


                        self.aalt[i] = self.actwpalt[i] # dial in altitude of next waypoint as calculated

                        t2go         = max(0.1,dist2wp)/max(0.01,self.gs[i])
                        self.avs[i]  = (self.actwpalt[i] - self.alt[i])/t2go
                       
                       
                if spd>0. and lnavon and self.swvnav[i]:
                    if spd<2.0:
                       self.actwpspd[i] = mach2cas(spd,self.alt[i])                            
                    else:    
                       self.actwpspd[i] = cas2tas(spd,self.alt[i])
                else:
                    self.actwpspd[i] = -999.

                # Calculate distance before waypoint where to start the turn
                # Turn radius:      R = V2 tan phi / g
                # Distance to turn: wpturn = R * tan (1/2 delhdg) but max 4 times radius
                turnrad = self.tas[i]*self.tas[i]/tan(self.bank[i]) /g0 /nm # [nm] default bank angle per flight phase
#                    print turnrad
                dy = (self.actwplat[i]-self.lat[i])
                dx = (self.actwplon[i]-self.lon[i])*cos(radians(self.lat[i]))
                qdr[i] = degrees(atan2(dx,dy))                    
                self.actwpturn[i] = max(3.,abs(turnrad*tan(radians(0.5*degto180(qdr[i]- \
                     self.route[i].wpdirfrom[self.route[i].iactwp])))))  # [nm]                
                
            # Set headings based on swlnav
            self.ahdg = np.where(self.swlnav, qdr, self.ahdg)

        # NOISE: Turbulence
        if self.turbulence:
            timescale=np.sqrt(simdt)
            trkrad=np.radians(self.trk)
            
            #write turbulences in array
            turb=np.array(self.standardturbulence)
            turb=np.where(turb>1e-6,turb,1e-6)
            
            #horizontal flight direction
            turbhf=np.random.normal(0,turb[0]*timescale,self.ntraf) #[m]
            
            #horizontal wing direction
            turbhw=np.random.normal(0,turb[1]*timescale,self.ntraf) #[m]
            
            #vertical direction
            turbalt=np.random.normal(0,turb[2]*timescale,self.ntraf) #[m]
            
            #latitudinal, longitudinal direction
            turblat=np.cos(trkrad)*turbhf-np.sin(trkrad)*turbhw #[m]
            turblon=np.sin(trkrad)*turbhf+np.cos(trkrad)*turbhw #[m]

        else:
            turbalt=np.zeros(self.ntraf) #[m]
            turblat=np.zeros(self.ntraf) #[m]
            turblon=np.zeros(self.ntraf) #[m]


        # ASAS AP switches
        #--------- Select Autopilot settings to follow: destination or ASAS ----------

        # desired autopilot settings due to ASAS
        self.deshdg = self.asasactive*self.asashdg + (1-self.asasactive)*self.ahdg
        self.desspd = self.asasactive*self.asasspd + (1-self.asasactive)*self.aspd
        self.desalt = self.asasactive*self.asasalt + (1-self.asasactive)*self.aalt
        self.desvs  = self.asasactive*self.asasvsp + (1-self.asasactive)*self.avs

        # check for the flight envelope
        self.perf.limits()

        # update autopilot settings with values within the flight envelope
        # speed
        self.aspd = (self.lspd ==0)*self.desspd + (self.lspd!=0)*self.lspd
        # altitude
        self.aalt = (self.lalt ==0)*self.desalt + (self.lalt!=0)*self.lalt
        # hdg
        self.ahdg = self.deshdg
        # vs
        self.avs = (self.lvs==0)*self.desvs + (self.lvs!=0)*self.lvs

        # below crossover altitude: CAS=const, above crossover altitude: MA = const
        #climb/descend above crossover: Ma = const, else CAS = const  
        #ama is fixed when above crossover
        check = self.abco*(self.ama == 0.)
        swma = np.where(check==True)
        self.ama[swma] = cas2mach(self.aspd[swma], self.alt[swma])
        # ama is deleted when below crossover
        check2 = self.belco*(self.ama!=0.)
        swma2 = np.where(check2==True)
        self.ama[swma2] = 0. 

        #---------- Basic Autopilot  modes ----------

        # SPD HOLD/SEL mode: aspd = autopilot selected speed (first only eas)
        # for information:    
        self.aptas = (self.actwpspd>0.)*self.actwpspd + \
                                  (self.actwpspd<=0.)*self.aptas
        self.delspd = self.aptas - self.tas 
        swspdsel = np.abs(self.delspd) > 0.4  # <1 kts = 0.514444 m/s
        ax = np.minimum(abs(self.delspd / max(1e-8,simdt)), self.ax)
        self.tas = swspdsel * (self.tas + ax * np.sign(self.delspd) *  \
                                          simdt) + (1. - swspdsel) * self.aptas
                                          
        # print "DELSPD", self.delspd/simdt, "AX", self.ax, "SELECTED", ax
        # without that part: non-accelerating ac would have TAS = 0            
        # print "1-sw", (1. - swspdsel) * self.aptas
        # print "NEW TAS", self.tas

        # Speed conversions
        self.cas = vtas2cas(self.tas, self.alt)
        self.gs  = self.tas
        self.M   = vtas2mach(self.tas, self.alt)

        # Update performance every self.perfdt seconds
        if abs(simt - self.perft0) >= self.perfdt:               
            self.perft0 = simt            
            self.perf.perf()

        # update altitude
        self.eps = np.array(self.ntraf * [0.01])  # almost zero for misc purposes
        swaltsel = np.abs(self.aalt-self.alt) >      \
                  np.maximum(3.,np.abs(2. * simdt * np.abs(self.vs))) # 3.[m] = 10 [ft] eps alt
#        print swaltsel

        self.vs = swaltsel*np.sign(self.aalt-self.alt)*       \
                    ( (1-self.swvnav)*np.abs(1500./60.*ft) +    \
                      self.swvnav*np.abs(self.avs)         )

        self.alt = swaltsel * (self.alt + self.vs * simdt) +   \
                   (1. - swaltsel) * self.aalt + turbalt

        # HDG HOLD/SEL mode: ahdg = ap selected heading
        delhdg = (self.ahdg - self.trk + 180.) % 360 - 180.  # [deg]

        # print delhdg
        # omega = np.degrees(g0 * np.tan(self.aphi) / \
        # np.maximum(self.tas, self.eps))

        # nominal bank angles per phase from BADA 3.12
        omega = np.degrees(g0 * np.tan(self.bank) / \
                           np.maximum(self.tas, self.eps))

        self.hdgsel = np.abs(delhdg) > np.abs(2. * simdt * omega)

        self.trk = (self.trk + simdt * omega * self.hdgsel * np.sign(delhdg)) % 360.

        #--------- Kinematics: update lat,lon,alt ----------
        ds = simdt * self.gs

        self.lat = self.lat + np.degrees(ds * np.cos(np.radians(self.trk)+turblat) \
                                         / Rearth)

        self.lon = self.lon + np.degrees(ds * np.sin(np.radians(self.trk)+turblon) \
                                         / np.cos(np.radians(self.lat)) / Rearth)

        # Update trails when switched on
        if self.swtrails:
            self.trails.update(simt, self.lat, self.lon,
                               self.lastlat, self.lastlon,
                               self.lasttim, self.id, self.trailcol)
        else:
            self.lastlat = self.lat
            self.lastlon = self.lon
            self.lattime = simt

        # ----------------AREA check----------------
        # Update area once per areadt seconds:
        if self.swarea and abs(simt - self.areat0) > self.areadt:
            # Update loop timer
            self.areat0 = simt
            # Check all aircraft
            i = 0
            while (i < self.ntraf):
                # Current status
                if self.area == "Square":
                    inside = self.arealat0 <= self.lat[i] <= self.arealat1 and \
                             self.arealon0 <= self.lon[i] <= self.arealon1 and \
                             self.alt[i] >= self.areafloor and \
                             (self.alt[i] >= 1500 or self.swtaxi)
                elif self.area == "Circle":
                    pass
                    # Average of lat
                    latavg = (radians(self.lat[i]) + radians(self.fir_circle_point[0])) / 2
                    cosdlat = (cos(latavg))

                    # Distance x to centroid
                    dx = (self.lon[i] - self.fir_circle_point[1]) * cosdlat * 60
                    dx2 = dx * dx

                    # Distance y to centroid
                    dy = self.lat[i] - self.fir_circle_point[0]
                    dy2 = dy * dy * 3600

                    # Radius squared
                    r2 = self.fir_circle_radius * self.fir_circle_radius

                    # Inside if smaller
                    inside = (dx2 + dy2) < r2

                # Compare with previous: when leaving area: delete command

                if self.inside[i] and not inside:
                    self.delete(self.id[i])

                else:
                    # Update area status
                    self.inside[i] = inside
                    i = i + 1

        return

    def id2idx(self, acid):
        """Find index of aircraft id"""
        try:
            return self.id.index(acid.upper())
        except:
            return -1

    def changeTrailColor(self, color, idx):
        """Change color of aircraft trail"""
        # print color
        # print idx
        # print "     " + str(self.trails.colorsOfAC[idx])
        self.trailcol[idx] = self.trails.colorList[color]
        # print "     " + str(self.trails.colorsOfAC[idx])
        return

    def setNoise(self, A):
        """Noise (turbulence, ADBS-transmission noise, ADSB-truncated effect)"""
        self.noise              = A
        self.trunctime          = 1                   # seconds
        self.transerror         = [1, 100, 100 * ft]  # [degree,m,m] standard bearing, distance, altitude error
        self.standardturbulence = [0, 0.1, 0.1]       # m/s standard turbulence  (nonnegative)
        # in (horizontal flight direction, horizontal wing direction, vertical)

        self.turbulence     = self.noise
        self.ADSBtransnoise = self.noise
        self.ADSBtrunc      = self.noise

    def engchange(self, acid, engid):
        """Change of engines"""
        self.perf.engchange(acid, engid)
        return

    def create(self, arglist):  # CRE command

        if len(arglist) < 7:
            return False

        acid  = arglist[0]

        if self.id.count(acid.upper()) > 0:
            return False, acid+" already exists"

        actype  = arglist[1]
        aclat   = arglist[2]
        aclon   = arglist[3]
        achdg   = arglist[4]
        acalt   = arglist[5]  # m
        cmdspd  = arglist[6]  # Mach/IAS kts (float)

        if 0.1 < cmdspd < 1.0 :
            acspd = mach2tas(cmdspd, acalt)
        else:
            acspd = cas2tas(cmdspd * kts, acalt)

        sw = self.create_ac(acid, actype, aclat, aclon, achdg, acalt, acspd)
        return sw

    def selhdg(self, arglist):  # HDG command

        # Select heading command: HDG acid, hdg

        if len(arglist) < 2:
            return False  # Error/Display helptext
        print "en voorbij de check"
        # unwrap arguments
        idx = arglist[0]  # aircraft index
        hdg = arglist[1]  # float

        # Give autopilot commands
        self.ahdg[idx]   = hdg
        self.swlnav[idx] = False
        # Everything went ok!
        return True
