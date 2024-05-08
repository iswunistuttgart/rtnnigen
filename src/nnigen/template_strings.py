
template_FB_inference = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1" ProductVersion="[[TWINCAT_VERSION]]">
  <POU Name="FB_[[NAME]]" Id="{[[UUID]]}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_[[NAME]]
VAR_INPUT
	pointer_input: POINTER TO [[DATA_TYPE]];
	pointer_output : POINTER TO [[DATA_TYPE]];
	
END_VAR
VAR	
  i : UINT;
  id : UINT;
  iq : UINT;
  flag_LoadWeights : BOOL := FALSE;
  flag_checkWeights : BOOL := FALSE;
  load_weights : FB_LoadWeights;
  filePath : T_MaxString := '[[filePath_weights]]';
  nn : [[NAME_ST_LAYERS]];
  hash_sha_256_twincat : ARRAY[0..3] OF LREAL;
  compare_res : DINT := 99;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[IF NOT flag_LoadWeights THEN
		load_weights(execute := TRUE,filePath := filePath,ReadAdr := ADR(nn.weights), ReadLen :=  SIZEOF(nn.weights));
		IF NOT load_weights.busy THEN 
			flag_LoadWeights := TRUE;
		END_IF
ELSIF NOT flag_checkWeights THEN
	F_GenerateHashValue(hashMode:=E_HashMode.HASH_SHA256,pData := ADR(nn.weights),nData := SIZEOF(nn.weights)-32,pHash := ADR(hash_sha_256_twincat),nHash:=32);
	compare_res := MEMCMp(pBuf1 := ADR(hash_sha_256_twincat),ADR(nn.weights.hash_sha_256),32);
	IF compare_res = 0 THEN
		flag_checkWeights := TRUE;
	END_IF
ELSE
	MEMCPY(destAddr:=ADR(nn.layer_input),srcAddr:=pointer_input,n:=SIZEOF([[DATA_TYPE]])*nn.layers[0].num_neurons);
  // input normalization
	IF nn.layers[0].normalization = act_type.normalization THEN
		FOR id := 0 TO nn.layers[0].num_neurons-1 DO
			 nn.layer_input[id] := F_normalization(x:=nn.layer_input[id],mean:=nn.weights.normalization_mean[id],std:=nn.weights.normalization_std[id]);
	 	END_FOR
	END_IF
  // forward inference
	FOR i := 0 TO nn.num_layers-2 DO
		F_ForwardPropagation(	layer_pre	:= 	nn.layers[i],
								layer_next	:=	nn.layers[i+1],
								pointer_in 	:=	ADR(nn.layer_input),
								pointer_out	:=	ADR(nn.layer_output)	);
		MEMCPY(destAddr:=ADR(nn.layer_input),srcAddr:=ADR(nn.layer_output),n:=SIZEOF([[DATA_TYPE]])*nn.layers[i+1].num_neurons);
	END_FOR
	MEMCPY(destAddr:=pointer_output,srcAddr:=ADR(nn.layer_output),n:=SIZEOF([[DATA_TYPE]])*nn.layers[SIZEOF(nn.layers)/SIZEOF(nn.layers[0])-1].num_neurons);
  // output denormalization
	IF nn.layers[SIZEOF(nn.layers)/SIZEOF(nn.layers[0])-1].normalization = act_type.denormalization THEN
		FOR iq := 0 TO nn.layers[SIZEOF(nn.layers)/SIZEOF(nn.layers[0])-1].num_neurons-1 DO
			 pointer_output[iq] := F_denormalization(x:=pointer_output[iq],mean:=nn.weights.denormalization_mean[iq],std:=nn.weights.denormalization_std[iq]);
	 	END_FOR
	 END_IF
END_IF
]]></ST>
    </Implementation>
    <LineIds Name="FB_[[NAME]]">
      <LineId Id="36" Count="0" />
      <LineId Id="54" Count="0" />
      <LineId Id="59" Count="1" />
      <LineId Id="57" Count="0" />
      <LineId Id="63" Count="0" />
      <LineId Id="44" Count="7" />
      <LineId Id="43" Count="0" />
      <LineId Id="38" Count="0" />
      <LineId Id="9" Count="0" />
    </LineIds>
  </POU>
</TcPlcObject>"""

template_DUT_Layers = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1" ProductVersion="[[TWINCAT_VERSION]]">
  <DUT Name="[[NAME_ST_LAYERS]]" Id="{[[UUID]]}">
    <Declaration><![CDATA[TYPE [[NAME_ST_LAYERS]]:
STRUCT
[[STRUCT_CONTENTS]]
END_STRUCT
END_TYPE
]]></Declaration>
  </DUT>
</TcPlcObject>
"""
template_DUT_LayersWeights = """
<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1" ProductVersion="[[TWINCAT_VERSION]]">
  <DUT Name="[[NAME_ST_LAYERS]]_LayerWeights" Id="{[[UUID]]}">
    <Declaration><![CDATA[TYPE [[NAME_ST_LAYERS]]_LayerWeights :
STRUCT
[[STRUCT_CONTENTS]]
END_STRUCT
END_TYPE
]]></Declaration>
  </DUT>
</TcPlcObject>
"""
