      COMMON /PCM_CAV/ OMEGA,RET,FRO,ALPHA(MXSP),RIN(MXSP),ICESPH,
     *                IPRPCM,IRETCAV,IPOLYG,AREATS
      COMMON /PCM_DAT/ EPS,EPSINF,DR,RSOLV,ICOMPCM,NPCMMT
      COMMON /PCM_CH/  QSN(MXTS),QSE(MXTS),QSENEQ(MXTS),QLOC(MXTS,3),
     *                PCMEN,PCMEE,PCMNE,PCMNN,QNUC,FN,FE
      COMMON /PCM_PLY/ XE(MXSP),YE(MXSP),ZE(MXSP),RE(MXSP),SSFE(MXSP),
     *                ISPHE(MXTS),STOT,VOL,NESF,NESFP,NC(30)
      COMMON /PCM_SPH/ INA(MXSP),IAN(MXCENT), IDXSPH(MXCENT),
     $                 RSPH(MXCENT)
      COMMON /PCM_TES/ CCX,CCY,CCZ,XTSCOR(MXTSPT),YTSCOR(MXTSPT),
     *                ZTSCOR(MXTSPT),AS(MXTS),RDIF,NVERT(MXTS),NTS,
     *                NTSIRR,NRWCAV
      COMMON /PCM_LU/  LUPCMD,LUCAVD,LUPCMI(8)
      COMMON /PCM_DER/ DERCEN(MXSP,MXCENT,3,3),DERRAD(MXSP,MXCENT,3)
      COMMON /PCM_SURPOT/ POTCAVNUC(MXTS), POTCAVELE(MXTS)
