#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3		// Use modern global access method and strict wave access.


//Master functions are:
//	DP_DeconvolveEX_DblExp
//	DP_DeconvolveEX_SingleExp
//Depending on the functional form of the instrument response function

//Written by Demetrios Pagonis
//demetrios.pagonis@colorado.edu


Function DP_Deconvolution()
	DisplayProcedure "DP_Deconvolution"
End

//--------------------------------------------

Function DP_Deconvolve_DblExp(wX, WY , wDest, Tau1, A1, Tau2, A2, NIter, SmoothError )

	//Deconvolution algorithm for DOUBLE EXPONENTIAL instrument function
	//Takes input data wX (time) and wY (signal), writes deconvolved output to wDest
	//Double-exponential instrument function defined by Tau1, A1, Tau2, A2
	//NIter: number of iterations of the deconvolution algorithm, 0 is autostop
	//SmoothError: number of time points to use for error smoothing. Set to 0 for no smoothing

	Wave wX
	Wave WY
	Wave /Z wDest

	Variable Tau1
	Variable A1
	Variable Tau2
	Variable A2
	Variable NIter
	Variable SmoothError
	Variable LastR2, R2=0.01
	Variable ForceIterations = 1

	Duplicate /O wY, wError, wConv
	Duplicate /FREE wY wLastConv
	Duplicate /FREE wX wX_free

	If(!WaveExists(wDest))
		Make /N=(numpnts(wY)) /O wDeconvolvedOutput
		Wave wDest = wDeconvolvedOutput
	EndIf

	If(NIter == 0)
		ForceIterations = 0
		NIter = 10
	EndIf

	Variable NPTS = numpnts(wDest)
	Variable ii
	Variable maxval, minval
	wDest = wY

	For( ii = 0 ; ii < NIter ; ii += 1)
		wLastConv=wConv
		DP_ConvolveXYDbleExp(wX_free, wDest, Tau1, A1, Tau2, A2, "wConv")
		wError = wConv - WY
		LastR2=R2
		R2=PCJ_NaNPearson(wConv, WY)^2
		//print R2
		If((abs(R2-LastR2)/LastR2)*100 > 1 || ForceIterations == 1)
			DP_AdjGuess(wDest, wError, SmoothError)
		Else
			print "Stopped deconv at N=",ii,"%R2 change =",(abs(R2-LastR2)/LastR2)*100
			ii = NIter
			wConv=wLastConv
		EndIf
		DoUpdate
	EndFor

	Return 0
End


//-------------------------------------------------------

Function DP_Deconvolve_SingleExp (wX, WY , wDest, Tau, NIter, SmoothError )

	//Deconvolution algorithm for SINGLE EXPONENTIAL instrument function
	//Takes input data wX (time) and wY (signal), writes deconvolved output to wDest
	//Tau: Single-exponential instrument function with timescale
	//NIter: number of iterations of the deconvolution algorithm, 0 is autostop
	//SmoothError: number of time points to use for error smoothing. Set to 0 for no smoothing


	Wave wX
	Wave WY
	Wave /Z wDest
	Variable NIter
	Variable Tau
	Variable SmoothError

	Variable LastR2, R2=0.01
	Variable ForceIterations = 1

	Duplicate /O wY, wError, wConv
	Duplicate /FREE wY wLastConv
	Duplicate /FREE wX wX_free

	If(!WaveExists(wDest))
		Make /N=(numpnts(wY)) /O wDeconvolvedOutput
		Wave wDest = wDeconvolvedOutput
	EndIf

	If(NIter == 0)
		ForceIterations = 0
		NIter = 20
	EndIf

	Variable NPTS = numpnts(wDest)
	Variable ii
	Variable maxval, minval
	wDest = wY

	For( ii = 0 ; ii < NIter ; ii += 1)
		wLastConv=wConv
		DP_ConvolveXY( wX_free, wDest, "wConv", Tau )
		wError = wConv - WY
		LastR2=R2
		R2=PCJ_NaNPearson(wConv, WY)^2
		//print R2
		If((abs(R2-LastR2)/LastR2)*100 > 1 || ForceIterations == 1)
			DP_AdjGuess(wDest, wError, SmoothError)
		Else
			print "Stopped deconv at N=",ii,"%R2 change =",(abs(R2-LastR2)/LastR2)*100
			ii = NIter
			wConv=wLastConv
		EndIf
		DoUpdate
	EndFor

	Return 0
End

//--------------------------------------------

Function DP_Deconvolve_CustomIRF(wX, WY , wDest, wIRF_x,wIRF_y, NIter, SmoothError )

	//Deconvolution algorithm for USER DEFINED instrument function
	//Takes input data wX (time) and wY (signal), writes deconvolved output to wDest
	//Instrument response function defined by wIRF_X and wIRF_y
	//IRF MUST START AT 1 AND GO TO 0 (i.e. depassivation IRF as defined in Deconv manuscript)
	//NIter: number of iterations of the deconvolution algorithm
	//SmoothError: number of time points to use for error smoothing. Set to 0 for no smoothing

	Wave wX
	Wave WY
	Wave /Z wDest
	Wave wIRF_x
	Wave wIRF_y

	Variable NIter
	Variable SmoothError
	Variable LastR2, R2=0.01
	Variable ForceIterations = 1

	Duplicate /O wY, wError, wConv
	Duplicate /FREE wY wLastConv
	Duplicate /FREE wX wX_free

	If(!WaveExists(wDest))
		Make /N=(numpnts(wY)) /O wDeconvolvedOutput
		Wave wDest = wDeconvolvedOutput
	EndIf

	If(NIter == 0)
		ForceIterations = 0
		NIter = 10
	EndIf

	Variable NPTS = numpnts(wDest)
	Variable ii
	Variable maxval, minval
	wDest = wY

	For( ii = 0 ; ii < NIter ; ii += 1)
		wLastConv=wConv
		DP_ConvolveXY_customIRF(wX_free, wDest, "wConv", wIRF_x,wIRF_y)
		wError = wConv - WY
		LastR2=R2
		R2=PCJ_NaNPearson(wConv, WY)^2
		//print R2
		If((abs(R2-LastR2)/LastR2)*100 > 1 || ForceIterations == 1)
			DP_AdjGuess(wDest, wError, SmoothError)
		Else
			print "Stopped deconv at N=",ii,"%R2 change =",(abs(R2-LastR2)/LastR2)*100
			ii = NIter
			wConv=wLastConv
		EndIf
		DoUpdate
	EndFor

	Return 0
End

//--------------------------------------------------------

Function DP_AdjGuess(wG, wE, NSmooth)

	Wave wG, wE
	Variable NSmooth

	If(NSmooth == 0 || NSmooth == 1)
		wG -= wE
	ElseIf(NSmooth > 1)
		Duplicate /FREE wE wE_Smooth
		Smooth /B NSmooth, wE_Smooth
		wG -= wE_Smooth
	Else
		Print "Error (DP_AdjGuess in DP_Deconvolution): NSmooth =",NSmooth
		Abort "DP_AdjGuess in DP_Deconvolution was passed a bad value for NSmooth"
	EndIf
	wG = (wG[p] < 0) ? 0 : wG[p]
	Return 0
End

//-------------------------------------------------------



Function DP_ConvolveXY( wx, wy, sW, TimeConst )
	//convolves XY data with a single-exponential defined by TimeConst
	//creates/overwrites a wave sW for output
	// function zero-pads start of the wave, one-pads the end and normalizes using those regions

	//Convolution:
	// (f*g)(t) = integral( f(tau)g(t-tau)dtau) over all time

	Wave wx,wy
	String sW
	Variable TimeConst

	Make /N=(numpnts(wX)-1) /FREE wDeltaX = wX[p+1]-wX[p]
	Variable tau
	Variable deltatau = wavemin(wDeltaX)/50

	Variable NPTS = numpnts(wY)
	Variable iPt
	Variable IntegralSum
	Variable xi

	Variable Term1, Term2, PtTerm1

	Make /N=(NPTS) /O $sW = NaN
	Wave wConvolvedData = $sW

	//Do the convolution
	For( iPt = 0 ; iPt < NPTS ; iPt += 1)	//for each point in the output wave (f*g)(t)

		xi = (iPt == NPTS-1) ? wX[iPt]+((wX[iPt]-wX[iPt-1])/2) : (wX[iPt]+wX[iPt+1])/2
		IntegralSum = 0

		For( tau = 0 ; tau <= 10*TimeConst ; tau += deltatau )		//integrate across tau

			If(tau <= xi )
				//Term1 = interp((xi-tau), wx, wy)		//g(t-tau) term	//switched to binarysearch 14-July 2020
				PtTerm1 = binarysearch(wX,xi-tau)
				Term1 = (PtTerm1 >=0) ? wY[PtTerm1] : 0	//zero when xi-tau is outside bounds of wx
				Term2 = (1/timeconst)*exp(-1*(tau)/TimeConst)		//f(tau) term
				IntegralSum += Term1*Term2*deltatau
			EndIf

		EndFor

		wConvolvedData[iPt] = IntegralSum
	EndFor

	Return 0
End

//-----------------------------------------------------------

Function DP_ConvolveXYDbleExp(wX, wY, Tau1, A1, Tau2, A2, sW)

	//Convolve XY data with a double exponential function with parameters
	// tau 1 and 2, relative intensities A1 and A2
	// create/overwrite destination wave sW

	Wave wX, wY
	Variable Tau1
	Variable Tau2
	Variable A1
	Variable A2
	String sW

	Make /N=(numpnts(wX)-1) /FREE wDeltaX = wX[p+1]-wX[p]
	Variable tau
	Variable deltatau = wavemin(wDeltaX)/50
	deltatau = (deltatau<0.01) ? 0.01 : deltatau

	Variable NPTS = numpnts(wY)
	Variable iPt
	Variable IntegralSum
	Variable xi
	Variable LongTau = (tau1 > tau2) ? tau1 : tau2

	Variable Term1, Term2, PtTerm1

	Make /N=(NPTS) /O $sW = NaN
	Wave wConvolvedData = $sW

	//Do the convolution
	For( iPt = 0 ; iPt < NPTS ; iPt += 1)	//for each point in the output wave (f*g)(t)

		xi = (iPt == NPTS-1) ? wX[iPt]+((wX[iPt]-wX[iPt-1])/2) : (wX[iPt]+wX[iPt+1])/2
		IntegralSum = 0

		For( tau = 0 ; tau <= 10*LongTau ; tau += deltatau )		//integrate across tau

			If(tau <= xi )
				//Term1 = interp((xi-tau), wx, wy)		//g(t-tau) term	//switched to binarysearch 14-July 2020
				PtTerm1 = binarysearch(wX,xi-tau)
				Term1 = (PtTerm1 >=0) ? wY[PtTerm1] : 0	//zero when xi-tau is outside bounds of wx
				Term2 = (A1/tau1) * exp(-1*tau/tau1) + (A2/tau2) * exp(-1*tau/tau2)		//f(tau) term
				IntegralSum += Term1*Term2*deltatau
			EndIf

		EndFor

		wConvolvedData[iPt] = IntegralSum
	EndFor

	Return 0
End

//-------------------------------------------------------



Function DP_ConvolveXY_customIRF( wx, wy, sW, wIRF_x,wIRF_y )
	//convolves XY data with a custom IRF
	//IRF MUST START AT 1 AND GO TO 0 (i.e. depassivation IRF as defined in Deconv manuscript)

	//creates/overwrites a wave sW for output

	//Convolution:
	// (f*g)(t) = integral( f(tau)g(t-tau)dtau) over all time

	Wave wx,wy
	String sW
	Wave wIRF_x, wIRF_y

	Make /N=(numpnts(wX)-1) /FREE wDeltaX = wX[p+1]-wX[p]
	Variable tau
	Variable deltatau =DP_GetValueFromPercentile(wDeltaX,10,0) //wavemin(wDeltaX)/50

	Variable NPTS = numpnts(wy)
	Variable IRFTime = ceil(wavemax(wIRF_x))
	Variable iPt
	Variable IntegralSum
	Variable xi

	Variable Term1, Term2, PtTerm1

	Duplicate /FREE wIRF_Y, wIRF_Y_DIFF
	Differentiate wIRF_Y/X=wIRF_X/D=wIRF_Y_DIFF
	wIRF_Y_DIFF *= -1

	Make /N=(NPTS) /O $sW = NaN
	Wave wConvolvedData = $sW

	//Do the convolution
	For( iPt = 0 ; iPt < NPTS ; iPt += 1)	//for each point in the output wave (f*g)(t)
		xi = (iPt == NPTS-1) ? wX[iPt]+((wX[iPt]-wX[iPt-1])/2) : (wX[iPt]+wX[iPt+1])/2
		IntegralSum = 0

		For( tau = 0 ; tau <= IRFTime ; tau += deltatau )		//integrate across tau

			If(tau <= xi )
				//Term1 = interp((xi-tau), wx, wy)		//g(t-tau) term	//switched to binarysearch 14-July 2020
				PtTerm1 = binarysearch(wx,xi-tau)
				Term1 = (PtTerm1 >=0) ? wy[PtTerm1] : 0	//zero when xi-tau is outside bounds of wx
				Term2 = interp(tau,wIRF_x, wIRF_Y_DIFF)		//f(tau) term
				IntegralSum += Term1*Term2*deltatau
			EndIf

		EndFor

		wConvolvedData[iPt] = IntegralSum
	EndFor

	Return 0
End

//--------------------------------------------------------------

Function DP_ConvFFT(wX,wY,wIRFx,wIRFy,sW)

	Wave wX
	Wave wY
	Wave wIRFx
	Wave wIRFy

	String sW

	Variable TimeStart=datetime

	Make /N=(numpnts(wX)-1) /FREE wXdelta=wX[p+1]-wX[p]
	Make /N=(numpnts(wIRFx)-1) /FREE wIRFXdelta=wIRFx[p+1]-wIRFx[p]
	Variable /G dX = (wavemin(wXdelta) > wavemin(wIRFXdelta)) ? 0.01*wavemin(wIRFXdelta) : 0.01*wavemin(wXdelta)
	Variable PowerdX = -1*floor(log(dX))
	dX = round(dX*10^Powerdx)/10^Powerdx


	Variable Len_wX = wX[numpnts(wX)-1] - wX[0]
	Variable Len_wIRFx = wIRFx[numpnts(wIRFx)-1] - wIRFx[0]
	Variable XStart = wX[0]
	Variable IRFXStart = wIRFx[0]

	//--------X waves starting from zero
	Make /N=(numpnts(wX)) /FREE wX_0Start
	wX_0Start = wX[p]-XStart

	Make /N=(numpnts(wIRFx)) /FREE wIRFX_0Start
	wIRFX_0Start = wIRFx[p]-IRFXStart

	//----------Make dIRF--------------
	Duplicate /O wIRFY, wdIRFY
	Differentiate wIRFy /X=wIRFx  /D=wdIRFy

	//----------Make the waves for FFT -- fixed scaling, same length -------------
	Variable NPTS = ceil(Len_wX/dX)
	NPTS += ceil(Len_wIRFx/dX)
	NPTS = NPTS+10-mod(NPTS,10) //divisible by 5 and 2 for speed

	Make /N=(NPTS) /O wY2, wdIRF2
	Setscale /P x,0,dX,wY2,wdIRF2
	wY2 = (BinarySearch(wX_0Start, x) >=0) ? wY[BinarySearch(wX_0Start, x)] : 0
	wdIRF2 = (BinarySearch(wIRFX_0Start, x) >=0) ? interp(x, wIRFX_0Start, wdIRFy ) : 0 //wdIRFy[BinarySearch(wIRFx,x)]

	//-------------Do the FFT, concolution, iFFT----------
	FFT /DEST=wY_FFT /Z wY2
	FFT /DEST=wdIRF_FFT /Z wdIRF2
	Make /N=(numpnts(wY_FFT)) /O /C wConv_FFT
	//Setscale /P x,0,1/Freq,wConv_FFT
	wConv_FFT= wY_FFT * wdIRF_FFT
	iFFT /Dest=wConv2 /Z wConv_FFT
	Setscale /P x,0,dX,wConv2
	wConv2 *=dX

	//convert back to native time base
	Redimension /N=(numpnts(wX)+1) wX_0Start
	wX_0Start[numpnts(wX)] = wX_0Start[numpnts(wX)-1]+wavemax(wXdelta)

	Make /N=(numpnts(wY)) /O wConv1
	wConv1 = mean(wConv2, wX_0Start[p], wX_0Start[p+1])

	Duplicate /O wConv1 $sW
	KillWaves /Z wConv1, wConv2, wdIRF2, wY2, wY_FFT, wdIRF_FFT,wConv_FFT


	Variable /G RunTime = datetime-timestart
	Return 0
End

//--------------------------------------------------------------------------------------

Function DP_Deconv_FFT_DblExp(wX, WY , wDest, Tau1, A1, Tau2, A2, NIter, SmoothError )

	Wave wX
	Wave WY
	Wave /Z wDest
	Variable Tau1
	Variable A1
	Variable Tau2
	Variable A2
	Variable NIter
	Variable SmoothError


	Variable FastTau = (Tau1 > Tau2) ? Tau2 : Tau1
	Variable SlowTau = (Tau1 > Tau2) ? Tau1 : Tau2
	Variable dX = FastTau/2
	Variable Len = SlowTau*10
	Variable NPTS = ceil(Len/dX)

	Make /N=(NPTS) /FREE wIRFx, wIRFy
	wIRFx = p*dX
	wIRFy = 1 - A1 * exp(wIRFx[p]*-1/Tau1) - A2 * exp(wIRFx[p]*-1/Tau2)

	DP_Deconv_FFT(wX, WY , wDest, wIRFx,wIRFy, NIter, SmoothError )

	Return 0
End

//------------------------------------------------------------------------------------

Function DP_Conv_FFT_DbleExp(wX, wY, Tau1, A1, Tau2, A2, sW)

	Wave wX, wY
	Variable Tau1
	Variable Tau2
	Variable A1
	Variable A2
	String sW

	Variable FastTau = (Tau1 > Tau2) ? Tau2 : Tau1
	Variable SlowTau = (Tau1 > Tau2) ? Tau1 : Tau2
	Variable dX = FastTau/2
	Variable Len = SlowTau*10
	Variable NPTS = ceil(Len/dX)

	Make /N=(NPTS) /FREE wIRFx, wIRFy
	wIRFx = p*dX
	wIRFy = 1 - A1 * exp(wIRFx[p]*-1/Tau1) - A2 * exp(wIRFx[p]*-1/Tau2)

	DP_ConvFFT(wX,wY,wIRFx,wIRFy,sW)

End

//------------------------------------------------------------------------------------

Function DP_Deconv_FFT(wX, WY , wDest, wIRF_x,wIRF_y, NIter, SmoothError )


	Wave wX
	Wave WY
	Wave /Z wDest
	Wave wIRF_x
	Wave wIRF_y

	Variable NIter
	Variable SmoothError
	Variable LastR2, R2=0.01
	Variable ForceIterations = 1

	Variable TimeStart=datetime


//-------------------Iterative solution below-------------------

//	Duplicate /O wY, wError, wConv
//	Duplicate /FREE wY wLastConv
//	Duplicate /FREE wX wX_free
//
//	If(!WaveExists(wDest))
//		Make /N=(numpnts(wY)) /O wDeconvolvedOutput
//		Wave wDest = wDeconvolvedOutput
//	EndIf
//
//	If(NIter == 0)
//		ForceIterations = 0
//		NIter = 10
//	EndIf
//
//	Variable NPTS = numpnts(wDest)
//	Variable ii
//	wDest = wY
//
//	For( ii = 0 ; ii < NIter ; ii += 1)
//		wLastConv=wConv
//		DP_ConvFFT(wX_free,wDest,wIRF_x,wIRF_y,"wConv")
//		wError = wConv - WY
//		LastR2=R2
//		R2=PCJ_NaNPearson(wConv, WY)^2
//		//print R2
//		If((abs(R2-LastR2)/LastR2)*100 > 1 || ForceIterations == 1)
//			DP_AdjGuess(wDest, wError, SmoothError)
//		Else
//			print "Stopped deconv at N=",ii,"%R2 change =",(abs(R2-LastR2)/LastR2)*100
//			ii = NIter
//			wConv=wLastConv
//		EndIf
//		DoUpdate
//	EndFor
//----------------------------end iterative solution

//-----------------------------------Direct solution below

	Make /N=(numpnts(wX)-1) /FREE wXdelta=wX[p+1]-wX[p]
	Make /N=(numpnts(wIRF_x)-1) /FREE wIRFXdelta=wIRF_x[p+1]-wIRF_x[p]
	Variable /G dX = (wavemin(wXdelta) > wavemin(wIRFXdelta)) ? 0.01*wavemin(wIRFXdelta) : 0.01*wavemin(wXdelta)
	Variable PowerdX = -1*floor(log(dX))
	dX = round(dX*10^Powerdx)/10^Powerdx

	Variable Len_wX = wX[numpnts(wX)-1] - wX[0]
	Variable Len_wIRFx = wIRF_x[numpnts(wIRF_x)-1] - wIRF_x[0]
	Variable XStart = wX[0]
	Variable IRFXStart = wIRF_x[0]

	//--------X waves starting from zero
	Make /N=(numpnts(wX)) /FREE wX_0Start
	wX_0Start = wX[p]-XStart

	Make /N=(numpnts(wIRF_x)) /FREE wIRFX_0Start
	wIRFX_0Start = wIRF_x[p]-IRFXStart

	//----------Make dIRF--------------
	Duplicate /O wIRF_Y, wdIRFY
	Differentiate wIRF_y /X=wIRF_x  /D=wdIRFy

	//----------Make the waves for FFT -- fixed scaling, same length -------------
	Variable NPTS = ceil(Len_wX/dX)
	NPTS += ceil(Len_wIRFx/dX)
	NPTS = NPTS+10-mod(NPTS,10) //divisible by 5 and 2 for speed

	Make /N=(NPTS) /O wY2, wdIRF2
	Setscale /P x,0,dX,wY2,wdIRF2
	//wY2 = (BinarySearch(wX_0Start, x) >=0) ? wY[BinarySearch(wX_0Start, x)] : 0
	wY2 = (BinarySearch(wX_0Start, x) >=0) ? interp(x, wX_0Start, wY ) : 0
	wdIRF2 = (BinarySearch(wIRFX_0Start, x) >=0) ? interp(x, wIRFX_0Start, wdIRFy ) : 0 //wdIRFy[BinarySearch(wIRFx,x)]

	//-------------Do the FFT, deconvolution, iFFT----------
	FFT /DEST=wY_FFT /Z wY2
	FFT /DEST=wdIRF_FFT /Z wdIRF2
	Make /N=(numpnts(wY_FFT)) /O /C wDeconv_FFT
	//Setscale /P x,0,1/Freq,wConv_FFT
	wDeconv_FFT= wY_FFT / wdIRF_FFT
	iFFT /Dest=wDeconv2 /Z wDeconv_FFT
	Setscale /P x,0,dX,wDeconv2
	wDeconv2 /=dX

	//convert back to native time base
	Redimension /N=(numpnts(wX)+1) wX_0Start
	wX_0Start[numpnts(wX)] = wX_0Start[numpnts(wX)-1]+wavemax(wXdelta)

	Make /N=(numpnts(wY)) /O wDeconv1
	wDeconv1 = mean(wDeconv2, wX_0Start[p], wX_0Start[p+1])

	Duplicate /O wDeconv1 wDest

//--------------------------------------end direct solution-------------


	KillWaves /Z wDeconv1, wDeconv2, wdIRF2, wY2, wY_FFT, wdIRF_FFT,wDeconv_FFT

	Variable /G RunTime = datetime-timestart
	Return 0
End


//----------------------------------------------------------------

Function DP_FitDblExp(wY,wX[,PtA,PtB,x0,x1,y0,y1,A1,tau1,tau2])

	//Fits a constrained double exponential where A1 + A2 = 1

	//Function will fit the data in the range set by optional parameters, either
	//in point space (PtA & PtB inputs) or x values (x0,x1 inputs). X inputs get priority
	//use x inputs when the depassivation starts between points


	//Default is to fit a double exponential going to zero. set parameter y0 to have y offset

	//Default is to normalize to the value at the start of the depassivation
	//Optional parameter y1 will normalize data to a different value (e.g. the passivated average)

	//Initial guesses for A1, tau1, tau2 are optional. defaults are 0.5, 1, and 20

	Wave wY, wX
	Variable PtA, PtB, x0, x1, y0, y1,A1,tau1,tau2

	If(ParamIsDefault(PtA))
		PtA = 0
	EndIf

	If(ParamIsDefault(PtB))
		PtB = numpnts(wX)-1
	EndIf

	If(ParamIsDefault(x0))
		x0=0
	Else
		PtA = ceil(BinarySearchInterp(wX,x0))
	EndIf

	If(!ParamIsDefault(x1))
		PtB = BinarySearchInterp(wX,x1)
	EndIf

	If(ParamIsDefault(y0))
		y0 = 0
	EndIf

	Variable NormFactor = wY[PtA]
	If(!ParamIsDefault(y1))
		NormFactor = y1
	EndIf

	If(ParamIsDefault(A1))
		A1=0.5
	EndIf

	If(ParamIsDefault(tau1))
		tau1=1
	EndIf

	If(ParamIsDefault(tau2))
		tau2=20
	EndIf

	//bookkeeping names, numpnts
	String sFit=nameofwave(wY)
	String sFitX = "fit_"+sFit+"_X"
	String sFitY = "fit_"+sFit+"_Y"
	Variable NPTS_Free = PtB - PtA + 1

	//Make the waves that will store the IRF
	Make /N=(NPTS_Free) /O $sFitX, $sFitY
	Wave wFitX = $sFitX
	Wave wFitY = $sFitY

	//make the waves that will be passed to FuncFit
	Make /N=(NPTS_Free) /FREE wY_norm, wX_shifted
	wX_shifted = wX[p+PtA] - wX[PtA]
	wY_norm = (wY[p+PtA]-y0) / (NormFactor-y0)

	//do the Fit
	Make/D/N=3/O W_coef
	W_coef[0] = {A1,tau1,tau2}
	FuncFit DP_DblExp_NormalizedIRF W_coef wY_norm /X=wX_shifted /D=wFitY

	//Scale the fit to y0 and y1
	wFitX=wX_shifted+x0
	wFitY = (wFitY[p] * (NormFactor-y0)) + y0

	Return 0
End

Function DP_DblExp_NormalizedIRF(w,x) : FitFunc
	Wave w
	Variable x

	//CurveFitDialog/ These comments were created by the Curve Fitting dialog. Altering them will
	//CurveFitDialog/ make the function less convenient to work with in the Curve Fitting dialog.
	//CurveFitDialog/ Equation:
	//CurveFitDialog/ f(x) = A1 * exp(-1*x/tau1) + (1-A1) * exp(-1*x/tau2)
	//CurveFitDialog/ End of Equation
	//CurveFitDialog/ Independent Variables 1
	//CurveFitDialog/ x
	//CurveFitDialog/ Coefficients 3
	//CurveFitDialog/ w[0] = A1
	//CurveFitDialog/ w[1] = tau1
	//CurveFitDialog/ w[2] = tau2

	return w[0] * exp(-1*x/w[1]) + (1-w[0]) * exp(-1*x/w[2])
End
