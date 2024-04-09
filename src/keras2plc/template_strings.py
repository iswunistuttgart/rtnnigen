
template_FB_NN_POU = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1" ProductVersion="[[TWINCAT_VERSION]]">
  <POU Name="FB_[[NAME]]" Id="{[[UUID]]}" SpecialFunc="None">
    <Declaration><![CDATA[FUNCTION_BLOCK FB_[[NAME]]
VAR_INPUT
	pointer_input: POINTER TO [[DATA_TYPE]];
	pointer_output : POINTER TO [[DATA_TYPE]];
	nn : Layers;
END_VAR
VAR	
	i : UINT;
END_VAR
]]></Declaration>
    <Implementation>
      <ST><![CDATA[MEMCPY(destAddr:=ADR(nn.layer_input),srcAddr:=pointer_input,n:=SIZEOF([[DATA_TYPE]])*nn.layers[0].num_neurons);
FOR i := 0 TO nn.num_layers-2 DO
	F_ForwardPropagation(	layer_pre	:= 	nn.layers[i],
							layer_next	:=	nn.layers[i+1],
							pointer_in 	:=	ADR(nn.layer_input),
							pointer_out	:=	ADR(nn.layer_output)	);
	MEMCPY(destAddr:=ADR(nn.layer_input),srcAddr:=ADR(nn.layer_output),n:=SIZEOF([[DATA_TYPE]])*nn.layers[i+1].num_neurons);
END_FOR
MEMCPY(destAddr:=pointer_output,srcAddr:=ADR(nn.layer_output),n:=SIZEOF([[DATA_TYPE]])*nn.layers[nn.num_layers-1].num_neurons);]]></ST>
    </Implementation>
    <LineIds Name="FB_[[NAME]]">
      <LineId Id="18" Count="7" />
      <LineId Id="9" Count="0" />
    </LineIds>
  </POU>
</TcPlcObject>"""

template_GVL_NN = """<?xml version="1.0" encoding="utf-8"?>
<TcPlcObject Version="1.1.0.1" ProductVersion="[[TWINCAT_VERSION]]">
  <GVL Name="GVL_[[NAME]]" Id="{[[UUID]]}">
    <Declaration><![CDATA[{attribute 'qualified_only'}
VAR_GLOBAL
	nn : [[NAME_ST_LAYERS]];
END_VAR]]></Declaration>
  </GVL>
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
