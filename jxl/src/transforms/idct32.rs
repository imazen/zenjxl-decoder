// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::excessive_precision)]

/// Apply 32-point IDCT to elements at `data[col + k*stride]` for k=0..32.
#[inline(always)]
pub(super) fn apply_idct_32(data: &mut [f32], col: usize, stride: usize) {
    let v0 = data[col];
    let v1 = data[col + stride];
    let v2 = data[col + 2 * stride];
    let v3 = data[col + 3 * stride];
    let v4 = data[col + 4 * stride];
    let v5 = data[col + 5 * stride];
    let v6 = data[col + 6 * stride];
    let v7 = data[col + 7 * stride];
    let v8 = data[col + 8 * stride];
    let v9 = data[col + 9 * stride];
    let v10 = data[col + 10 * stride];
    let v11 = data[col + 11 * stride];
    let v12 = data[col + 12 * stride];
    let v13 = data[col + 13 * stride];
    let v14 = data[col + 14 * stride];
    let v15 = data[col + 15 * stride];
    let v16 = data[col + 16 * stride];
    let v17 = data[col + 17 * stride];
    let v18 = data[col + 18 * stride];
    let v19 = data[col + 19 * stride];
    let v20 = data[col + 20 * stride];
    let v21 = data[col + 21 * stride];
    let v22 = data[col + 22 * stride];
    let v23 = data[col + 23 * stride];
    let v24 = data[col + 24 * stride];
    let v25 = data[col + 25 * stride];
    let v26 = data[col + 26 * stride];
    let v27 = data[col + 27 * stride];
    let v28 = data[col + 28 * stride];
    let v29 = data[col + 29 * stride];
    let v30 = data[col + 30 * stride];
    let v31 = data[col + 31 * stride];

    // Stage 1: 2-point butterfly on indices {0,16}
    let v32 = v0 + v16;
    let v33 = v0 - v16;
    let v34 = v8 + v24;
    let v35 = v8 * std::f32::consts::SQRT_2;
    let v36 = v35 + v34;
    let v37 = v35 - v34;
    let v38 = v36.mul_add(0.5411961001461970, v32);
    let v39 = (-v36).mul_add(0.5411961001461970, v32);
    let v40 = v37.mul_add(1.3065629648763764, v33);
    let v41 = (-v37).mul_add(1.3065629648763764, v33);

    // Stage 2: 4-point butterfly on indices {4,12,20,28}
    let v42 = v4 + v12;
    let v43 = v12 + v20;
    let v44 = v20 + v28;
    let v45 = v4 * std::f32::consts::SQRT_2;
    let v46 = v45 + v43;
    let v47 = v45 - v43;
    let v48 = v42 + v44;
    let v49 = v42 * std::f32::consts::SQRT_2;
    let v50 = v49 + v48;
    let v51 = v49 - v48;
    let v52 = v50.mul_add(0.5411961001461970, v46);
    let v53 = (-v50).mul_add(0.5411961001461970, v46);
    let v54 = v51.mul_add(1.3065629648763764, v47);
    let v55 = (-v51).mul_add(1.3065629648763764, v47);

    // Combine stages 1+2
    let v56 = v52.mul_add(0.5097955791041592, v38);
    let v57 = (-v52).mul_add(0.5097955791041592, v38);
    let v58 = v54.mul_add(0.6013448869350453, v40);
    let v59 = (-v54).mul_add(0.6013448869350453, v40);
    let v60 = v55.mul_add(0.8999762231364156, v41);
    let v61 = (-v55).mul_add(0.8999762231364156, v41);
    let v62 = v53.mul_add(2.5629154477415055, v39);
    let v63 = (-v53).mul_add(2.5629154477415055, v39);

    // Stage 3: 8-point butterfly on even indices {2,6,10,14,18,22,26,30}
    let v64 = v2 + v6;
    let v65 = v6 + v10;
    let v66 = v10 + v14;
    let v67 = v14 + v18;
    let v68 = v18 + v22;
    let v69 = v22 + v26;
    let v70 = v26 + v30;
    let v71 = v2 * std::f32::consts::SQRT_2;
    let v72 = v71 + v67;
    let v73 = v71 - v67;
    let v74 = v65 + v69;
    let v75 = v65 * std::f32::consts::SQRT_2;
    let v76 = v75 + v74;
    let v77 = v75 - v74;
    let v78 = v76.mul_add(0.5411961001461970, v72);
    let v79 = (-v76).mul_add(0.5411961001461970, v72);
    let v80 = v77.mul_add(1.3065629648763764, v73);
    let v81 = (-v77).mul_add(1.3065629648763764, v73);
    let v82 = v64 + v66;
    let v83 = v66 + v68;
    let v84 = v68 + v70;
    let v85 = v64 * std::f32::consts::SQRT_2;
    let v86 = v85 + v83;
    let v87 = v85 - v83;
    let v88 = v82 + v84;
    let v89 = v82 * std::f32::consts::SQRT_2;
    let v90 = v89 + v88;
    let v91 = v89 - v88;
    let v92 = v90.mul_add(0.5411961001461970, v86);
    let v93 = (-v90).mul_add(0.5411961001461970, v86);
    let v94 = v91.mul_add(1.3065629648763764, v87);
    let v95 = (-v91).mul_add(1.3065629648763764, v87);
    let v96 = v92.mul_add(0.5097955791041592, v78);
    let v97 = (-v92).mul_add(0.5097955791041592, v78);
    let v98 = v94.mul_add(0.6013448869350453, v80);
    let v99 = (-v94).mul_add(0.6013448869350453, v80);
    let v100 = v95.mul_add(0.8999762231364156, v81);
    let v101 = (-v95).mul_add(0.8999762231364156, v81);
    let v102 = v93.mul_add(2.5629154477415055, v79);
    let v103 = (-v93).mul_add(2.5629154477415055, v79);

    // Combine even stages
    let v104 = v96.mul_add(0.5024192861881557, v56);
    let v105 = (-v96).mul_add(0.5024192861881557, v56);
    let v106 = v98.mul_add(0.5224986149396889, v58);
    let v107 = (-v98).mul_add(0.5224986149396889, v58);
    let v108 = v100.mul_add(0.5669440348163577, v60);
    let v109 = (-v100).mul_add(0.5669440348163577, v60);
    let v110 = v102.mul_add(0.6468217833599901, v62);
    let v111 = (-v102).mul_add(0.6468217833599901, v62);
    let v112 = v103.mul_add(0.7881546234512502, v63);
    let v113 = (-v103).mul_add(0.7881546234512502, v63);
    let v114 = v101.mul_add(1.0606776859903471, v61);
    let v115 = (-v101).mul_add(1.0606776859903471, v61);
    let v116 = v99.mul_add(1.7224470982383342, v59);
    let v117 = (-v99).mul_add(1.7224470982383342, v59);
    let v118 = v97.mul_add(5.1011486186891553, v57);
    let v119 = (-v97).mul_add(5.1011486186891553, v57);

    // Stage 4: 16-point butterfly on odd indices {1,3,5,...,31}
    let v120 = v1 + v3;
    let v121 = v3 + v5;
    let v122 = v5 + v7;
    let v123 = v7 + v9;
    let v124 = v9 + v11;
    let v125 = v11 + v13;
    let v126 = v13 + v15;
    let v127 = v15 + v17;
    let v128 = v17 + v19;
    let v129 = v19 + v21;
    let v130 = v21 + v23;
    let v131 = v23 + v25;
    let v132 = v25 + v27;
    let v133 = v27 + v29;
    let v134 = v29 + v31;
    let v135 = v1 * std::f32::consts::SQRT_2;
    let v136 = v135 + v127;
    let v137 = v135 - v127;
    let v138 = v123 + v131;
    let v139 = v123 * std::f32::consts::SQRT_2;
    let v140 = v139 + v138;
    let v141 = v139 - v138;
    let v142 = v140.mul_add(0.5411961001461970, v136);
    let v143 = (-v140).mul_add(0.5411961001461970, v136);
    let v144 = v141.mul_add(1.3065629648763764, v137);
    let v145 = (-v141).mul_add(1.3065629648763764, v137);
    let v146 = v121 + v125;
    let v147 = v125 + v129;
    let v148 = v129 + v133;
    let v149 = v121 * std::f32::consts::SQRT_2;
    let v150 = v149 + v147;
    let v151 = v149 - v147;
    let v152 = v146 + v148;
    let v153 = v146 * std::f32::consts::SQRT_2;
    let v154 = v153 + v152;
    let v155 = v153 - v152;
    let v156 = v154.mul_add(0.5411961001461970, v150);
    let v157 = (-v154).mul_add(0.5411961001461970, v150);
    let v158 = v155.mul_add(1.3065629648763764, v151);
    let v159 = (-v155).mul_add(1.3065629648763764, v151);
    let v160 = v156.mul_add(0.5097955791041592, v142);
    let v161 = (-v156).mul_add(0.5097955791041592, v142);
    let v162 = v158.mul_add(0.6013448869350453, v144);
    let v163 = (-v158).mul_add(0.6013448869350453, v144);
    let v164 = v159.mul_add(0.8999762231364156, v145);
    let v165 = (-v159).mul_add(0.8999762231364156, v145);
    let v166 = v157.mul_add(2.5629154477415055, v143);
    let v167 = (-v157).mul_add(2.5629154477415055, v143);

    let v168 = v120 + v122;
    let v169 = v122 + v124;
    let v170 = v124 + v126;
    let v171 = v126 + v128;
    let v172 = v128 + v130;
    let v173 = v130 + v132;
    let v174 = v132 + v134;
    let v175 = v120 * std::f32::consts::SQRT_2;
    let v176 = v175 + v171;
    let v177 = v175 - v171;
    let v178 = v169 + v173;
    let v179 = v169 * std::f32::consts::SQRT_2;
    let v180 = v179 + v178;
    let v181 = v179 - v178;
    let v182 = v180.mul_add(0.5411961001461970, v176);
    let v183 = (-v180).mul_add(0.5411961001461970, v176);
    let v184 = v181.mul_add(1.3065629648763764, v177);
    let v185 = (-v181).mul_add(1.3065629648763764, v177);
    let v186 = v168 + v170;
    let v187 = v170 + v172;
    let v188 = v172 + v174;
    let v189 = v168 * std::f32::consts::SQRT_2;
    let v190 = v189 + v187;
    let v191 = v189 - v187;
    let v192 = v186 + v188;
    let v193 = v186 * std::f32::consts::SQRT_2;
    let v194 = v193 + v192;
    let v195 = v193 - v192;
    let v196 = v194.mul_add(0.5411961001461970, v190);
    let v197 = (-v194).mul_add(0.5411961001461970, v190);
    let v198 = v195.mul_add(1.3065629648763764, v191);
    let v199 = (-v195).mul_add(1.3065629648763764, v191);
    let v200 = v196.mul_add(0.5097955791041592, v182);
    let v201 = (-v196).mul_add(0.5097955791041592, v182);
    let v202 = v198.mul_add(0.6013448869350453, v184);
    let v203 = (-v198).mul_add(0.6013448869350453, v184);
    let v204 = v199.mul_add(0.8999762231364156, v185);
    let v205 = (-v199).mul_add(0.8999762231364156, v185);
    let v206 = v197.mul_add(2.5629154477415055, v183);
    let v207 = (-v197).mul_add(2.5629154477415055, v183);

    // Combine odd sub-stages
    let v208 = v200.mul_add(0.5024192861881557, v160);
    let v209 = (-v200).mul_add(0.5024192861881557, v160);
    let v210 = v202.mul_add(0.5224986149396889, v162);
    let v211 = (-v202).mul_add(0.5224986149396889, v162);
    let v212 = v204.mul_add(0.5669440348163577, v164);
    let v213 = (-v204).mul_add(0.5669440348163577, v164);
    let v214 = v206.mul_add(0.6468217833599901, v166);
    let v215 = (-v206).mul_add(0.6468217833599901, v166);
    let v216 = v207.mul_add(0.7881546234512502, v167);
    let v217 = (-v207).mul_add(0.7881546234512502, v167);
    let v218 = v205.mul_add(1.0606776859903471, v165);
    let v219 = (-v205).mul_add(1.0606776859903471, v165);
    let v220 = v203.mul_add(1.7224470982383342, v163);
    let v221 = (-v203).mul_add(1.7224470982383342, v163);
    let v222 = v201.mul_add(5.1011486186891553, v161);
    let v223 = (-v201).mul_add(5.1011486186891553, v161);

    // Final combination: even + odd
    let v224 = v208.mul_add(0.5006029982351963, v104);
    let v225 = (-v208).mul_add(0.5006029982351963, v104);
    let v226 = v210.mul_add(0.5054709598975436, v106);
    let v227 = (-v210).mul_add(0.5054709598975436, v106);
    let v228 = v212.mul_add(0.5154473099226246, v108);
    let v229 = (-v212).mul_add(0.5154473099226246, v108);
    let v230 = v214.mul_add(0.5310425910897841, v110);
    let v231 = (-v214).mul_add(0.5310425910897841, v110);
    let v232 = v216.mul_add(0.5531038960344445, v112);
    let v233 = (-v216).mul_add(0.5531038960344445, v112);
    let v234 = v218.mul_add(0.5829349682061339, v114);
    let v235 = (-v218).mul_add(0.5829349682061339, v114);
    let v236 = v220.mul_add(0.6225041230356648, v116);
    let v237 = (-v220).mul_add(0.6225041230356648, v116);
    let v238 = v222.mul_add(0.6748083414550057, v118);
    let v239 = (-v222).mul_add(0.6748083414550057, v118);
    let v240 = v223.mul_add(0.7445362710022986, v119);
    let v241 = (-v223).mul_add(0.7445362710022986, v119);
    let v242 = v221.mul_add(0.8393496454155268, v117);
    let v243 = (-v221).mul_add(0.8393496454155268, v117);
    let v244 = v219.mul_add(0.9725682378619608, v115);
    let v245 = (-v219).mul_add(0.9725682378619608, v115);
    let v246 = v217.mul_add(1.1694399334328847, v113);
    let v247 = (-v217).mul_add(1.1694399334328847, v113);
    let v248 = v215.mul_add(1.4841646163141662, v111);
    let v249 = (-v215).mul_add(1.4841646163141662, v111);
    let v250 = v213.mul_add(2.0577810099534108, v109);
    let v251 = (-v213).mul_add(2.0577810099534108, v109);
    let v252 = v211.mul_add(3.4076084184687190, v107);
    let v253 = (-v211).mul_add(3.4076084184687190, v107);
    let v254 = v209.mul_add(10.1900081235480329, v105);
    let v255 = (-v209).mul_add(10.1900081235480329, v105);

    // Store outputs in order from return tuple:
    // (v224, v226, v228, v230, v232, v234, v236, v238,
    //  v240, v242, v244, v246, v248, v250, v252, v254,
    //  v255, v253, v251, v249, v247, v245, v243, v241,
    //  v239, v237, v235, v233, v231, v229, v227, v225)
    data[col] = v224;
    data[col + stride] = v226;
    data[col + 2 * stride] = v228;
    data[col + 3 * stride] = v230;
    data[col + 4 * stride] = v232;
    data[col + 5 * stride] = v234;
    data[col + 6 * stride] = v236;
    data[col + 7 * stride] = v238;
    data[col + 8 * stride] = v240;
    data[col + 9 * stride] = v242;
    data[col + 10 * stride] = v244;
    data[col + 11 * stride] = v246;
    data[col + 12 * stride] = v248;
    data[col + 13 * stride] = v250;
    data[col + 14 * stride] = v252;
    data[col + 15 * stride] = v254;
    data[col + 16 * stride] = v255;
    data[col + 17 * stride] = v253;
    data[col + 18 * stride] = v251;
    data[col + 19 * stride] = v249;
    data[col + 20 * stride] = v247;
    data[col + 21 * stride] = v245;
    data[col + 22 * stride] = v243;
    data[col + 23 * stride] = v241;
    data[col + 24 * stride] = v239;
    data[col + 25 * stride] = v237;
    data[col + 26 * stride] = v235;
    data[col + 27 * stride] = v233;
    data[col + 28 * stride] = v231;
    data[col + 29 * stride] = v229;
    data[col + 30 * stride] = v227;
    data[col + 31 * stride] = v225;
}
