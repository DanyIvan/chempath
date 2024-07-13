    
import numpy as np

def three_body_rate(A0, AI, CN, CM, temp, den):
    B0 = A0*(300./temp)**CN
    BI = AI*(300./temp)**CM
    Y = np.log10(B0*den/BI)
    X = 1./(1. + Y**2)
    rate = B0*den/(1. + B0*den/BI) * 0.6**X
    return rate

def weird_rates(reactants, products, temp, den):
    ''' Cacluate rates of reactions that do not follow the a two body or
    three body rates. The rate equations were taken from the PhotochemPy model:
    https://github.com/Nicholaswogan/PhotochemPy
    '''
    # H2 + O -> OH + H
    if reactants ==['H2', 'O']:
        return  1.34E-15*(temp/298.)**6.52 *np.exp(-1460./temp)   # Low T fit NIST 05
    # endif

    # HO2 + HO2 -> H2O2 + O2
    elif reactants ==['HO2','HO2']:
    #JPL-02
        return  2.3E-13*np.exp(590./temp) + 1.7E-33*np.exp(1000./temp)*den
    # endif

    # O + O + M -> O2 + M
    elif reactants ==['O','O']:
    #note - re-hard coding into S+S->S2 evil I know...
        return  9.46E-34*np.exp(480./temp)*den  # NIST 05 low Temp in N2.  Its 2x bigger in H2 or O2
    # endif

    # CO + OH -> CO2 + H (also CS + HS -> CS2 + H)  #SORG
    elif reactants ==['CO','OH'] or reactants == ['CS','HS']:
        PATM = den*1.38E-16*temp/1.013E6   #a good pressure
        return  1.5E-13 * (1. + 0.6*PATM)     #JPL-02
    # endif

    #   CO + O + M -> CO2 + M
    elif reactants ==['CO','O']:
        return  2.2E-33*np.exp(-1780./temp)*den  # I use NIST 05 for 257-277 K
    # endif

    #gna - updated according to e-g 2011
    #   H + CO + M -> HCO + M (also H + CS + M -> HCS + M) #sorg
    elif reactants ==['H','CO'] or \
        reactants == ['H','CS']:
        return  2.0E-33*np.exp(-850./temp)*den  # I use NIST 05 for 333-1000 K, theory
    # endif

    #   H2CO + H -> H2 + HCO
    elif reactants ==['H2CO','H']:
        return  2.14E-12*(temp/298.)**1.62*np.exp(-1090./temp)    # NIST 2005 Baulch et al 2002
    # endif

    #   H + H + M -> H2 + M
    elif reactants ==['H','H']:
        return  8.85E-33*(temp/287)**(-0.6) * den  #gna Baluch 1994
    # endif

    #   H + OH + M -> H2O + M
    elif reactants ==['H','OH']:
        return  6.9E-31*(298./temp)**2 *den     # Baulch et al 2002 in N2
    # endif

    #   CH3 + CH3 + M  ->  C2H6 + M
    elif reactants ==['CH3','CH3']:
        A71_3 = 1.17e-25*np.exp(-500./temp)
        # if (background_spec .EQ. 'CO2') A71_3=A71_3*2.5    #CO2 rather than N2
        return  three_body_rate(A71_3,3.0E-11,3.75E0,1.0E0,temp,den)   # what a mess NIST 2005 -
    # return  1.7E-17/temp**2.3 * den
    # endif


    #   CH3 + H2CO  ->  CH4 + HCO
    elif reactants ==['CH3','H2CO']:
        return  4.9E-15*(temp/298.)**4.4 *np.exp(-2450./temp)    #NIST 2005
    # endif

    #   H + NO + M  ->  HNO + M
    elif reactants ==['H','NO']:
        A77_3 = 1.2E-31*np.exp(-210./temp)                       # there was a bug here - one that mattered
        return  three_body_rate(A77_3,2.4E-10, 1.17E0,0.41E0,temp,den)   # taken from NIST 2005 Tsang/Hampson 91
    # return  1.2E-31*(298./temp)**1.17 *np.exp(-210./temp) * den
    # endif

    #   N + N + M  ->  N2 + M
    elif reactants ==['N','N']:
    # return  8.3E-34*np.exp(500./temp) * den
    # return  1.25E-32                   #NIST 2005
        return  1.25E-32* den       #NIST 2005 BUG HERE - SHOULD BE X DEN (KEEP AS WEIRD)
    # endif


    #   HNO3 + OH  ->  H2O + NO3
    elif reactants ==['HNO3','OH']:
        AK0 = 2.4E-14*np.exp(460./temp)
        AK2 = 2.7E-17*np.exp(2199./temp)
        AK3M = 6.5E-34*np.exp(1335./temp)*den
        return  AK0 + AK3M/(1. + AK3M/AK2)  #JPL-06
    # endif

    #   H + HNO  ->  H2 + NO
    elif reactants ==['H','HNO']:
        return  3.0E-11 * np.exp(-500./temp)   # NIST 2005  ??? shouldn't this be 2body? (check on rx list)
    # endif

    #   C2H6 + O  ->  C2H5 + OH
    elif reactants ==['C2H6','O']:
        return  8.54E-12*(temp/300.)**1.5 *np.exp(-2920./temp)   # NIST 05
    # endif

    #   SO + O -> SO2
    elif reactants ==['SO','O']:
        return  6.0E-31 * den         # NIST 2005 updated from e-G 2011 gna
    # endif

    #    S + S -> S2
    elif reactants ==['S','S']:
    # return  1.2e-29 * den   # in H2S, but its much 1e4 slower in Ar
    # return  min([5.e-11, 3* A(JOO_O2,I)])      # reported rate is 3X larger for S+S in Ar than for O+O in Ar

        return  1.87E-33 * np.exp(-206/temp)*den #updated from e-G 2011 gna
    # endif

    #   S + S2 + M -> S3 + M
    elif reactants ==['S','S2']:
        return  min([5.0E-11, 2.5E-30 * den / 5.E+1])    # NIST reported In CO2 with factor 5 error
    # endif

    #    S2 + S2 + M -> S4 + M
    elif reactants ==['S2','S2']:
        return  min([5.0E-11,2.5E-30*den/1.E+1])    # NIST reported In CO2 with factor 5 error I'm taking lower bound
    # endif

    #     S + S3 + M -> S4 + M
    elif reactants ==['S','S3']:
        return  min([5.0E-11,2.5E-30*den/5.E+1])    # NIST reported In CO2 with factor 5 error
    #assumed rate equal to the S+S2 rate
    # endif

    #     S4 + S4 + M -> S8 + M
    #       OR
    #     S4 + S4 -> S8AER
    elif reactants ==['S4','S4']:
        return  min([5.0E-11,2.5E-30*den/1.E1])    # NIST reported In CO2 with factor 5 error I'm taking lower bound
    #assumed rate equal to S2+S2
    # endif

    #     S + CO + M -> OCS + M
    elif reactants ==['S','CO']:
        return  1.*2.2E-33*np.exp(-1780./temp)*den  # no information
    #  I'm guessing that S + CO goes at about the same rate as O + CO
    #  with the same activation energy and a 3X bigger A factor
    #-mc but yet factor of 1 out from, so this is the same as o+co

        #gna - DG 2011 has:
        #return  6.5E-33*np.exp(-2180./temp)*den  # assumed same as (CO+O)

    # endif

    #    OCS + S + M -> OCS2 + M   # NIST 8.3e-33*Den in Ar
    elif reactants ==['OCS','S']:
        return  8.3E-33*den     # reduce by 1000 to turn it off
    # endif


    #    CO + O1D + M -> CO2 + M
    elif reactants ==['CO','O1D']:
        return  0.0 * 8.00E-11           # reported but no sign of density dependence
    # endif

    #    O1D + N2 + M -> N2O + M   #this is really damn slow...
    elif reactants ==['O1D','N2']:
        return  2.80E-36*den*(temp/300.)**(-0.9) #JPL-06  no high pressure limit reported
    # endif

    #   CL + O2 + M -> CLOO + M    #IUPAC 2007 rec - only a low density rate (could modify TBDY to deal with this...)
    elif reactants ==['CL','O2']:
        return  1.4E-33*den*(temp/300.)**(-3.9)    #measured in N2
    # endif


    #   CL + NO + M -> CLNO + M
    elif reactants ==['CL','NO']:
        return  7.6E-32 * (300./temp)**1.8 * den #JPL-06 (no high pressure limit given)
    # endif

    #   CCL3NO4 + M -> CCL3O2 + NO2               WEIRD     #IUPAC-07
    elif reactants ==['CCL3NO4','M']:
        k0 =4.3E-3*np.exp(-10235/temp)*den
        kinf=4.8E16*np.exp(-11820/temp)
        Fc=0.32
        return  ((k0*kinf)/(k0 + kinf)* 10**(np.log10(Fc)/ \
    (1 +(np.log10(k0/kinf)/(0.75-1.27*np.log10(Fc)))**2)))/den
    # return  4.8e16*np.exp(-11820/temp)     #old
    # endif

    #   NO + CH3O -> HNO + H2CO              WEIRD     #IUPAC worksheet - 2 body with the 300/T form... (yuk 123)
    elif reactants ==['NO','CH3O']:
        return  2.3E-12 * (300./temp)**0.7
    # endif

    #   NO2 + CH3O + M -> CH3ONO2 + M  IUPAC
    elif reactants ==['NO2','CH3O']:
        k0 =(8.1E-29*(300./temp)**4.5)*den
        kinf=2.1E-11
        Fc=0.44
        return  ((k0*kinf)/(k0 + kinf)* 10**(np.log10(Fc)/ \
    (1 +(np.log10(k0/kinf)/(0.75-1.27*np.log10(Fc)))**2)))/den
    # endif


    #   NO2 + CH3O2 + M -> CH3O2NO2 +M                   WEIRD     # IUAPC-06 (yuk 131)
    elif reactants ==['NO2','CH3O2']:
        k0 =(2.5E-30*(300./temp)**5.5)*den
        kinf=1.8E-11
        Fc=0.36
        return  ((k0*kinf)/(k0 + kinf)* 10**(np.log10(Fc)/ \
    (1 +(np.log10(k0/kinf)/(0.75-1.27*np.log10(Fc)))**2)))/den
    # endif

    #   CH3O2NO2  M -> CH3O2 + NO2               WEIRD     #IUPAC
    elif reactants ==['CH3O2NO2','M']:
        k0 =9.0E-5*np.exp(-9690./temp)*den
        kinf=1.1E16*np.exp(-10560./temp)
        Fc=0.6
        return  ((k0*kinf)/(k0 + kinf)* 10**(np.log10(Fc)/ \
    (1 +(np.log10(k0/kinf)/(0.75-1.27*np.log10(Fc)))**2)))/den
    # endif


    #   CH2O2 + M -> CO + H2O               WEIRD     # IUPAC-06 thermal decomp
    elif reactants ==['CH2O2','M']:
        # first order rate constant of 6e4
        return 6E4/den #because it will be multiplied by den so net is 6e4*[CH2O2]
        # return 9.96E-20 #old version
        # endif


    #   CH2OOH + M -> OH + H2CO              WEIRD     #Vaghjinai et al. 1989 (via NIST) as 1st order -(yuk 278) has 1e-10 as 2body
    elif reactants ==['CH2OOH','M']:
        return 5E4/den #1st order rate, will be multiplied by den so net is 5e4*[CH2OOH]
        # return  1.e-10   #yuks rate
        # endif


    #    N2O5 + M -> NO3 + NO2               WEIRD   #IUPAC
    elif reactants ==['N2O5','M']:
        k0 =1.3E-3*np.exp(-11000./temp)*(300./temp)**3.5*den
        kinf=9.7E14*np.exp(-11080./temp)*(temp/300.)**0.1
        Fc=0.35
        return  ((k0*kinf)/(k0 + kinf)* 10**(np.log10(Fc)/ \
    (1 +(np.log10(k0/kinf)/(0.75-1.27*np.log10(Fc)))**2)))/den
    # endif


    #   HO2NO2 + M -> HO2 + NO2
    elif reactants ==['HO2NO2','M']:
        k0 =4.1E-5*np.exp(-10650./temp)*den
        kinf=4.8E15*np.exp(-11170./temp)     #there was an error here. should be -11170
        Fc=0.6                             #this should be Fc=0.6
        return  ((k0*kinf)/(k0 + kinf)* 10**(np.log10(Fc)/ \
    (1 +(np.log10(k0/kinf)/(0.75-1.27*np.log10(Fc)))**2)))/den
    # endif

    #   CL2O2 + M -> CLO + CLO
    elif reactants ==['CL2O2','M']:
        k0 =3.7E-7*np.exp(-7690./temp)*den
        kinf=7.9E15*np.exp(-8820./temp)
        Fc=0.45
        # return  (k0*kinf)/(k0 + kinf)* 10**(np.log10(Fc)/
        # $  (1 +(np.log10(k0/kinf)/(0.75-1.27*np.log10(Fc)))**2))

        return  ((k0*kinf)/(k0 + kinf)* 10**(np.log10(Fc)/ \
        (1 +(np.log10(k0/kinf)/(0.75-1.27*np.log10(Fc)))**2)))/den
    # endif

    #   CH3ONO + M -> CH3O + NO                WEIRD     1.06e15     -18282.                # Fernandez-Ramos et al. 1998 (300K-1500K) 1st order reaction
    elif reactants ==['CH3ONO','M']:
        return  1.06e15*np.exp(-18041./temp)#/den  #or is it first order?
        # return  1.06e15*np.exp(-18282./temp)  #old
    # endif


    #   CLO + O2 + M ->  CLO3 + M #  WEIRD     9.00E-28  (214 yuk) - THIS IS NOT IN JPL-06 ???
    elif reactants ==['CLO','O2'] and 'CLO3' in products:
        # CHEMJ(3,J).EQ.'CLO3']:
        A362_0=9.00E-28*temp**(-2.0)
        A362_inf=4.50E-7*temp**(-2.0)

        #tuning
        A362_0=9.00E-38*temp**(-2.0)
        A362_inf=4.50E-19*temp**(-2.0)

    #rego - from DeMore 1990
    # c      A362_0=1.00e-32*temp**(-2.0)
    # c      A362_inf=5.00e-12*temp**(-2.0)



    # c           return  three_body_rate(A362_0,A362_inf, 0.0E0,0.0E0,temp,den)   # from yuk (214)

    # c temptest
        return  0.0   #zeroing for now. eventually rebuild this as CLO.O2

    # c           return  three_body_rate(1.0e-32,5.0E-12, 2.0E0,2.0E0,temp,den)   # from yuk (214)
    # endif

    #   CLO + CLO3 + M -> CL2O4 +M              WEIRD     #Xu and Lin 2003
    #   CLO + CLO3  -> CLOO + OCLO              WEIRD     #Xu and Lin 2003 (slow)
    #   CLO + CLO3  -> OCLO +OCLO               WEIRD     #Xu and Lin 2003 (slow)
    elif reactants == ['CLO', 'CLO3'] and 'CL2O4' in products:
        # if (CHEMJ(3,J).EQ.'CL2O4') then
            A0=8.62E15*np.exp(-1826./temp)*temp**(-9.75)
            Ainf=1.43E-10*np.exp(-82./temp)*temp**(0.094)
            return three_body_rate(A0,Ainf, 0.e0,0.e0,temp,den)

    elif reactants == ['CLO', 'CLO3'] and 'CLOO' in products:
        return 1.85E-18*temp**2.28*np.exp(-2417./temp)

    elif reactants == ['CLO', 'CLO3','OCLO'] and 'OCLO' in products:
        return 1.42E-18*temp**2.11*np.exp(-2870./temp)
                
         


    #    O3 + CL + M -> CLO3                        WEIRD     #Simonaitis 1975
    elif reactants ==['O3','CL']:
        return   3E-30*den     # rate constant estimated for 300K (some words about t-dependence - could look to analogs...)
    # c           return   1E-31*den     #lowering this uncertain rate by a factor of 30 1e-13 is nonp.minial case
    # endif




    #    CLO + O2 -> CLOOO                       WEIRD     # DeMore 1990 rate (Via Shindell)
    elif reactants ==['CLO','O2'] and 'CLOOO' in products:
        # CHEMJ(3,J).EQ.'CLOOO']:
        #from DeMore 1990      #could take this to classic 3body
        A366_0=1.00E-36    #going for -3
        A366_inf=5.00E-12#
        return  three_body_rate(A366_0,A366_inf, 2.0e0,2.0e0,temp,den)
    # endif

    #    CLOOO + M -> CLO + O2                WEIRD - thermal decomp (divide above by eq rate in Prassad) - Vogel has 3.1e-18 at 200K
    elif reactants ==['CLOOO','M']:
        A367_EQ = 2.9E-26*np.exp(3500./temp)
        return  three_body_rate(A366_0,A366_inf, 2.0e0,2.0e0,temp,den) \
        /A367_EQ/1E5/den
    #    note use of coefficients from formation reaction so this is a dependency
    # this species was used a test in Catling et al. 2010 for the CLO.O2 adduct - see sensitivity analysis there...
    # endif

    #   O + CLO + M  ->  OCLO + M
    elif reactants ==['O','CLO']:
        Alow=8.60E-21*temp**(-4.1)*np.exp(-420./temp)
        Ainf=4.33E-11*temp**(-0.03)*np.exp(43./temp)

        return  three_body_rate(Alow,Ainf,0.0e0,0.0e0,temp,den)   # taken from Zhu and Lin 2003

    # endif


    #   OH + CLO3 + M  ->  HCLO4 or HO2 + OCLO  (Zhu and Lin 2001) - note this will interfere if we go back to Simonitatis#
    elif reactants ==['OH','CLO3'] and 'HCLO4' in products:
    # if (CHEMJ(3,J).EQ.'HCLO4']:
        A0=1.94E36*temp**(-15.3)*np.exp(-5542./temp)
        Ainf=3.2E-10*temp**(0.07)*np.exp(-25./temp)

        return  three_body_rate(A0,Ainf,0.0e0,0.0e0,temp,den)   # taken from Zhu and Lin 2001
        #should perhaps put the Simonaitis rate in here for completeness so we don't have to switch back and forth on reactions.rx
    # endif

    elif reactants ==['OH','CLO3'] and 'HO2' in products:
        return 2.1E-10*temp**(0.09)*np.exp(-18./temp)   #ditto


    #   CS2 + S  ->  CS + S2  #SORG
    elif reactants ==['CS2','S']:
    # Woiki et al 1995
        return  1.9E-14 * np.exp(-580./temp) * (temp/300.)**3.97
    # endif

    #   C2H6S + H ->  CH3SH + CH3  #SORG
    #3 needed because there is another C2H6S + H  2 body reaction in the table
    elif reactants ==['C2H6S','H','CH3SH']:
        # CHEMJ(3,J).EQ.'CH3SH']:


    #gna - possible mistake here: in D.-G. et al 2011  C2H6S + H ->  H2 + C2H4 + HS is listed with different reaction rate coefficients
    #  8.34E-12    -2212.     1.6
    # can't parse the Zhang paper so not sure what is correct.
        return  4.81E-12 * np.exp(-1100./temp) * (temp/300.)**1.70    # theory - Zhang et al. 2005
    # endif

    #this used to be same as  C2H6S + H ->  CH3SH + CH3  #SORG but that is not what shawn has in 2011 paper table
    #   C2H6S + H ->  H2 + C2H4 + HS #HC   Theory. Zhang et al. [2005]. Produces C2H5S, which then can split into C2H4 + HS
    elif reactants ==['C2H6S','H'] and 'H2' in products:
        # CHEMJ(3,J).EQ.'H2']:
        return  8.34E-12 * np.exp(-2212./temp) * (temp/300.)**1.60 # theory - Zhang et al. 2005
    # endif


    #gna (this was missing from sorg reactions.rx too so had to add it)
    #C2H6S + O -> CH3 + CH3 + SO
    #CH3SH + O -> CH3 + HSO
    #in reactions list as "WEIRD" but wasn't here...
    elif (reactants ==['C2H6S','O'] and 'CH3' in products):
    # # CHEMJ(3,J).EQ.'CH3') or \
    # ['C2H6SH','O']
    # CHEMJ(3,J).EQ.'CH3') )THEN
        return  1.30E-11 * np.exp(-410./temp) * (temp/298.)**1.1    # Sander 2006
    # endif

    #gna
    #C2H6S2 + O -> CH3 + CH3S + SO
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['C2H6S2','O'] and 'CH3' in products:
    # CHEMJ(3,J).EQ.'CH3') )THEN
        return  3.90E-11 * np.exp(290./temp) * (temp/298.)**1.1    # Sander 2006
    # endif

    #gna
    #C2H6S + OH -> CH21 + CH3S + CS2O
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['C2H6S','OH'] and 'CH21' in products:
    # CHEMJ(3,J).EQ.'CH21') )THEN
        return  1.10E-11 * np.exp(400./temp) * (temp/298.)**1.1    # Sander 2006
    # endif

    #gna
    #C2H6S2 + OH -> CH3 + CH3SH + S
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['C2H6S2','OH'] and 'CH3' in products:
    # CHEMJ(3,J).EQ.'CH3') )THEN
        return  6.00E-11 * np.exp(400./temp) * (temp/298.)**1.2    # Sander 2006
    # endif

    #gna
    #C2H6S + O --> CH3 + Ch3 + SO
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['C2H6S2','OH'] and 'CH3' in products:
    # CHEMJ(3,J).EQ.'CH3') )THEN
        return  6.00E-11 * np.exp(400./temp) * (temp/298.)**1.2    # Sander 2006
    # endif

    #gna
    #CH3 + OH -> CH3O + H
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['CH3','OH'] and'CH3O' in products:
    # CHEMJ(3,J).EQ.'CH3O') )THEN
        return  9.3E-11 * np.exp(-1606/temp) * (temp/298.)**1.    # Jasper 2007
    # endif

    #gna
    #CH3 + HNO -> CH4 + NO
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['CH3','HNO'] and 'CH4' in products:
    # CHEMJ(3,J).EQ.'CH4') )THEN
        return  1.85E-11 * np.exp(-176/temp) * (temp/298.)**0.6    # Choi and Lin 2005
    # endif


    #gna
    #H2S + H -> H2 + HS
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['H2S','H'] and 'H2' in products:
    # CHEMJ(3,J).EQ.'H2') )THEN
        return  3.66E-12 * np.exp(-455/temp) * (temp/298.)**1.94    # Choi and Lin 2005
    # endif

    #gna
    #SO + HCO -> HSO + CO
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['SO','HCO'] and 'HSO' in products:
    # CHEMJ(3,J).EQ.'HSO') )THEN
        return  5.6E-12  * (temp/298.)**0.4    # Kasting 1990
    # endif

    #gna
    #NH2 + H -> NH3
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['NH2','H'] and 'NH3' in products:
    # CHEMJ(3,J).EQ.'NH3') )THEN
        return  (6.e-30*den)/(1.+3.e-20*den)    # Gordon 1971
    # endif

    #gna
    #NH + H -> NH2
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['NH','H'] and 'NH2' in products:
    # CHEMJ(3,J).EQ.'NH2') )THEN
        return   (6.e-30*den)/(1.+3.e-20*den)   # Kasting 1982
    # endif

    #gna
    #CS + HS -> CS2 + H
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['CS','CS'] and 'CS2' in products:
    # CHEMJ(3,J).EQ.'CS2') )THEN
        return   1.5E-13*(1.+0.6*den)   # assumed samed as k(CO+OH)
    # endif

    #gna
    #CH3SH + OH --> CH3S + H2O
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['CH3SH','OH'] and 'CH3S' in products:
    # CHEMJ(3,J).EQ.'CH3S') )THEN
        return  9.90E-12 * np.exp(360/temp) * (temp/298.)**1.07    # Sander 2006
    # endif


    #gna
    #HCO + M --> H + CO + M
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['HCO','M'] and 'H' in products:
    # CHEMJ(3,J).EQ.'H') )THEN
        return  6.0E-11 * np.exp(-7721/temp) * den   #Krasnoperov et al 2004
    # endif

    #gna
    #HNO + M --> NO + H + M
    #in reactions list as "WEIRD" but wasn't here...
    elif reactants ==['HNO','M'] and 'NO' in products:
    # CHEMJ(3,J).EQ.'NO') )THEN
        return  1.04E-6 * np.exp(28618/temp)*(temp/298.)**(-1.61)*den #Tsang 1986
    # endif


    #gna -- ordering was wrong of CHEMJ indices compared to reactions.rx for sorg template

    #   CH3S + HCS ->CS + CH3SH #SORG
    elif reactants ==['HCS', 'CH3S']:

        return  1.18E-12*np.exp(-910./temp)*(temp/300.)**0.65  #Liu et al. 2006 (via NIST)

    # endif

    #   C + H2 -> CH23
    elif reactants ==['C','H2']:
        B0 = 8.75E-31 * np.exp(524./temp)
        BI = 8.3E-11
        return  B0*BI*den/(B0*den + BI)
    # endif

    #   CH + H2 -> CH3            #apparantly the same as C+H2->CH23
    elif reactants ==['CH','H2'] and 'CH3' in products:
        B0 = 8.75E-31 * np.exp(524./temp)
        BI = 8.3E-11
        return  B0*BI*den/(B0*den + BI)
    # endif


    #   CH23 + H -> CH3
    elif reactants ==['CH23','H'] and 'CH3' in products:
        B0 = 3.1E-30 * np.exp(457./temp)
        BI = 1.5E-10
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  C2H + H -> C2H2
    elif reactants ==['C2H','H']:
        B0 = 1.26E-18 * np.exp(-721./temp) / temp**3.1
        BI = 3.e-10
        return  B0*BI*den/(B0*den + BI)

    #gna - shawn has diff:
    #B0 = 2.64E-26 * np.exp(-721./temp) / (temp/300.)**3.1
    #BI = 3.e-10
    #return  B0*BI*den/(B0*den + BI)
    # endif

    #  CH23 + CO -> CH2CO

    elif reactants ==['CH23','CO']:
        B0 = 1.e-28
        BI = 1.e-15
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  CH3 + CO -> CH3CO
    elif reactants ==['CH3','CO']:
        return  1.4E-32 * np.exp(-3000./temp)*den
    # endif

    #  C2H2 + H -> C2H3
    elif reactants ==['C2H2','H']:
        B0 = 2.6E-31
        BI = 3.8E-11 * np.exp(-1374./temp) #gna shawn has 8.3E-11 in paper
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  C2H3 + CH4 ->  C2H4 + CH3
    elif reactants ==['C2H3','CH4']:
        return  2.4E-24 * temp**4.02 * np.exp(-2754./temp)
    # endif

    #  C2H4 + H -> C2H5
    #  C3H6 + H -> C3H7
    elif reactants ==['C2H4','H'] or reactants == ['C3H6','H']:
        B0 = 2.15E-29 * np.exp(-349./temp)
        BI = 4.95E-11 * np.exp(-1051./temp)
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  CH + CH4 -> C2H4 + H
    elif reactants ==['CH','CH4']:
        A270A = 2.5E-11 * np.exp(200./temp)
        A270B = 1.7E-10
        return  min([A270A,A270B])
    # endif

    #  C2H5 + CH3 -> C2H4 + CH4
    elif reactants ==['C2H5','CH3'] and 'C2H4'  in products:
        # .AND.CHEMJ(3,J).EQ.'C2H4']:
        return  3.25E-11/temp**0.5
    # endif

    #  C2H2 + OH -> CH2CO + H
    elif reactants ==['C2H2','OH'] and 'CH2CO' in products:
    # .EQ. 'CH2CO']:
        B0 = 5.8E-31 * np.exp(1258./temp)
        BI = 1.4E-12 * np.exp(388./temp)
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  C2H5 + CH3 -> C3H8 #gna shawn has diff
    elif reactants ==['C2H5','CH3'] and 'CH2CO' in products:
        # .AND.CHEMJ(3,J).EQ.CH2CO]:
        B0 = 2.519E-16 / temp**2.458
        BI = 8.12E-10 / temp**0.5
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  C3H8 + O -> C3H7 + OH
    elif reactants ==['C3H8','O']:
        return 1.6E-11 * np.exp(-2900./temp) + 2.2E-11 * np.exp(-2200./temp)
    # endif

    # C2H3 + CH3 -> C3H6
    elif reactants ==['C2H3','CH3']:
        B0 = 1.3E-22
        BI = 1.2E-10
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  CH + C2H4 -> CH2CCH2 + H                 WEIRD           # Romani et al. [1993]
    #  CH + C2H4 -> CH3C2H + H                 WEIRD                                                            # Romani et al. [1993]
    #this should be OK to capture both channels and proceed them at the same rate (which is what shawn did)
    elif reactants ==['CH','C2H4']:
        A272A = 5.5E-11 * np.exp(173./temp)
        A272B = 3.55E-10
        return  min([A272A,A272B])
    # endif

    #  CH2CCH2 + H -> CH3 + C2H2              WEIRD                                                            # Yung et al. [1984]
    #  CH2CCH2 + H -> C3H5                        WEIRD                                                            # Yung et al. [1984]
    elif reactants ==['CH2CCH2','H']:
        B0 = 8.e-24/temp**2 * np.exp(-1225./temp)
        if 'CH3' in products:
            BI=9.7E-13 * np.exp(-1550./temp)
        elif 'C3H5' in products:
            BI=1.4E-11 * np.exp(-1000./temp)

        return  B0*BI*den/(B0*den + BI)
    # endif

    #  C2H3 + C2H5 -> CH3 + C3H5              WEIRD                                                            # Romani et al. [1993]
    elif reactants ==['C2H3','C2H5'] and 'CH3' in products:
    # .EQ.'CH3']:
        B0 = 1.9E-27
        BI = 2.5E-11
        return  BI - B0*BI*den/(B0*den + BI)
    # endif


    #  C3H5 + H -> C3H6                        WEIRD                                                            # Yung et al. [1984] #has another branch#
    elif reactants ==['C3H5','H'] and 'C3H6' in products:
    # .EQ.'C3H6']:
        B0 = 1.e-28
        BI = 1.e-11
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  CH + C2H2 -> C3H2 + H                 WEIRD                                                            # Romani et al. [1993]
    elif reactants ==['CH','C2H2']:
        A271A = 1.75E-10 * np.exp(61./temp)
        A271B = 5.3E-10
        return  min([A271A,A271B])
    # endif

    #  CH23 + C2H2 -> CH3C2H                      WEIRD                                                            # Laufer et al. [1983] and Laufer [1981]
    elif reactants ==['CH23','C2H2'] and 'CH3C2H' in products:
        # .AND.CHEMJ(3,J).EQ.'CH3C2H']:
        B0 = 3.8E-25
        BI = 2.2E-12
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  CH3C2H + H -> CH3 + C2H2              WEIRD                                                            # Whytock et al [1976] and Von Wagner and Zellner [1972]
    #  CH3C2H + H -> C3H5                        WEIRD                                                            # Yung et al. [1984], same as RXN 303
    elif reactants ==['CH3C2H','H']:
        B0 = 8.e-24/temp**2 * np.exp(-1225./temp)
        BI = 9.7E-12 * np.exp(-1550./temp)
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  C3H2 + H -> C3H3                        WEIRD                                                            # Yung et al. [1984]
    #  C3H3 + H -> CH3C2H                      WEIRD                                                            # Yung et al. [1984], same as RXN 37
    #  C3H3 + H -> CH2CCH2                     WEIRD                                                            # Yung et al. [1984], same as RXN 337
    elif reactants ==['C3H2','H'] or reactants == ['C3H3','H']:
        B0 = 1.7E-26
        BI = 1.5E-10
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  CH23 + C2H2 -> CH2CCH2                     WEIRD                                                            # Laufer et al. [1983] and Laufer [1981]
    elif reactants ==['CH23','C2H2'] and 'CH2CCH2' in products:
        # .AND.CHEMJ(3,J).EQ.'CH2CCH2']:
        B0 = 3.8E-25
        BI = 3.7E-12
        return  B0*BI*den/(B0*den + BI)
    # endif

    #  C2H5 + H -> C2H6                        WEIRD                                                            # Gladstone [1983]
    elif reactants ==['C2H5','H'] and 'C2H6' in products:
        # .EQ.'C2H6']:
        B0 = 5.5E-23/temp**2 * np.exp(-1040./temp)
        BI = 1.5E-13 * np.exp(-440./temp)
        return  B0*BI*den/(B0*den + BI)
    # endif

    # added by nick
    #   H2CN + NH2  ->  HCN + NH3
    elif reactants ==['H2CN','NH2']:
        return  5.42E-11*(300/temp)**1.06 * np.exp(-60.8/temp)
    # endif

    #   H2CN + M  ->  HCN + H + M
    elif reactants ==['H2CN','M'] and 'HCN' in products:
    # .EQ.'HCN']:
        return  5.50E-17*den
    # endif

    #   HNCO + OH  ->  NCO + H2O
    elif reactants ==['HNCO','OH'] and 'NCO' in products:
    # .EQ.'NCO']:
        return  0.9 * 6.1E-17 * temp**1.5 * np.exp(-1809./temp)
    # endif

    #   HNCO + OH  ->  CO2 + NH2
    elif reactants ==['HNCO','OH'] and 'CO2' in products:
    # .EQ.'CO2']:
        return  0.1 * 6.1E-17 * temp**1.5 * np.exp(-1809./temp)
    # endif

    #   HNCO + H  ->  HCNOH
    elif reactants ==['HNCO','H'] and 'HCNOH' in products:
    # .EQ.'HCNOH']:
        PATM = den*1.38E-16*temp/1.000E6   #a good pressure in bar
        return (PATM/1.)*1.36E-8*(temp/298)**(-1.90)*np.exp(-1390./temp)
    # endif

    #   HNCO + H  ->  NH2 + CO
    elif reactants ==['HNCO','H'] and 'NH2' in products:
    # .EQ.'NH2']:
        return  8.63E-14 * (temp/298)**2.49 * np.exp(-1181./temp)
    # endif

    #   HNCO + H  ->  NCO + H2
    elif reactants ==['HNCO','H'] and 'NCO' in products:
    # .EQ.'NCO']:
        return  2.67E-13 * (temp/298)**2.41 * np.exp(-6194./temp)
    # endif

    #   HNCO + O  ->  CO2 + NH
    elif reactants ==['HNCO','O'] and 'CO2' in products:
    # .EQ.'CO2']:
        return  5.01E-13 * (temp/298)**1.41 * np.exp(-4292./temp)
    # endif

    #   HNCO + O  ->  NCO + OH
    elif reactants ==['HNCO','O'] and 'NCO' in products:
    # .EQ.'NCO']:
        return  6.08E-13 * (temp/298)**2.11 * np.exp(-5753./temp)
    # endif

    #   NCO + O  ->  CN + O2
    elif reactants ==['NCO','O'] and 'CN' in products:
    # .EQ.'CN']:
        return  4.05E-10 * (temp/298)**(-1.43) * np.exp(-3502./temp)
    # endif

    #   NCO + O  ->  CO + NO
    elif reactants ==['NCO','O'] and 'CO' in products:
    # .EQ.'CO']:
        return  6.5E-11 * (temp/298)**(-1.14)
    # endif

    #   NCO + H2  ->  HNCO + H
    elif reactants ==['NCO','H2'] and 'HNCO' in products:
    # .EQ.'HNCO']:
        return  6.54E-14 * temp**2.58 * np.exp(-2700./temp)
    # endif

    #   NCO + OH  ->  HNCO + O
    elif reactants ==['NCO','OH'] and 'HNCO' in products:
    # .EQ.'HNCO']:
        return  5.38E-14 * (temp/298)**2.27 * np.exp(-497./temp)
    # endif

    #   CN + C2H6  ->  HCN + C2H5
    elif reactants ==['CN','C2H6'] and 'HCN' in products:
    # .EQ.'HCN']:
        return  2.08E-11 * (temp/298.)**0.22 * np.exp(57.8/temp)
    # endif

    #   CN + CH4  ->  HCN + CH3
    elif reactants ==['CN','CH4'] and 'HCN' in products:
    # .EQ.'HCN']:
        return  5.11E-13 * (temp/298)**2.64 * np.exp(150./temp)
    # endif

    #   NO + C  ->  CO + N
    elif reactants ==['NO','C'] and 'CO' in products:
    # .EQ.'CO']:
        return  3.49E-11 * (temp/298)**(-0.02)
    # endif

    #   NH + N  ->  N2 + H
    elif reactants ==['NH','N'] and 'N2' in products:
    # .EQ.'N2']:
        return  1.95e-11 *(temp/298.)**0.51 * np.exp(-9.63/temp)
    # endif

    #   HCN + O  ->  NH + CO
    elif reactants ==['HCN','O'] and 'NH' in products:
    # .EQ.'NH']:
        return  8.88e-13 *(temp/298.)**1.21 * np.exp(-3851./temp)
    # endif

    #   HCN + O  ->  CN + OH
    elif reactants ==['HCN','O'] and 'CN' in products:
    # .EQ.'CN']:
        return  1.43e-12 *(temp/298.)**1.47 * np.exp(-3801./temp)
    else:
        raise Exception("WEIRD reaction is not in the program.")
