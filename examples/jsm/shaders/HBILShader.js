import {
	DataTexture,
	Matrix4,
	RepeatWrapping,
	Vector2,
	Vector3,
	Vector4,
} from 'three';

/**
 * References:
 * - https://drive.google.com/file/d/1fmceYuM5J2s8puNHZ9o4OF3YjqzIvmRR/view
 * - https://arxiv.org/pdf/2301.11376.pdf
 * - https://www.shadertoy.com/view/dsGBzW
 * - https://www.reddit.com/r/GraphicsProgramming/comments/17k4hpr/screen_space_horizon_gi/
 */

const HBILShader = {

	name: 'HBILShader',

	defines: {
		PERSPECTIVE_CAMERA: 1,
		SAMPLES: 16,
		SAMPLE_VECTORS: generateAoSampleKernelInitializer( 16 ),
		NORMAL_VECTOR_TYPE: 1,
		DEPTH_SWIZZLING: 'x',
		CLIP_RANGE_CHECK: 1,
		DISTANCE_FALL_OFF: 1,
		NV_ALIGNED_SAMPLES: 1,
		SCREEN_SPACE_RADIUS: 0,
		SCREEN_SPACE_RADIUS_SCALE: 100.0,
	},

	uniforms: {
		tDiffuse: { value: null },
		tNormal: { value: null },
		tDepth: { value: null },
		tNoise: { value: null },
		resolution: { value: new Vector2() },
		cameraNear: { value: null },
		cameraFar: { value: null },
		cameraProjectionMatrix: { value: new Matrix4() },
		cameraProjectionMatrixInverse: { value: new Matrix4() },
		radius: { value: 1. },
		distanceExponent: { value: 1. },
		thickness: { value: 1. },
		bias: { value: 0.001 },
		scale: { value: 1. },
	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;

			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`

		varying vec2 vUv;
		uniform sampler2D tDiffuse;
		uniform highp sampler2D tNormal;
		uniform highp sampler2D tDepth;
		uniform sampler2D tNoise;
		uniform vec2 resolution;
		uniform float cameraNear;
		uniform float cameraFar;
		uniform mat4 cameraProjectionMatrix;
		uniform mat4 cameraProjectionMatrixInverse;		
		uniform float radius;
		uniform float distanceExponent;
		uniform float thickness;
		uniform float bias;
		uniform float scale;
		
		#include <common>
		#include <packing>

		#ifndef FRAGMENT_OUTPUT
		#define FRAGMENT_OUTPUT vec4(vec3(ao), 1.)
		#endif

		const vec4 sampleKernel[SAMPLES] = SAMPLE_VECTORS;

		vec3 getViewPosition(const in vec2 screenPosition, const in float depth) {
			vec4 clipSpacePosition = vec4(vec3(screenPosition, depth) * 2.0 - 1.0, 1.0);
			vec4 viewSpacePosition = cameraProjectionMatrixInverse * clipSpacePosition;
			return viewSpacePosition.xyz / viewSpacePosition.w;
		}

		float getDepth(const vec2 uv) {  
			return textureLod(tDepth, uv.xy, 0.0).DEPTH_SWIZZLING;
		}

		float fetchDepth(const ivec2 uv) {   
			return texelFetch(tDepth, uv.xy, 0).DEPTH_SWIZZLING;
		}

		float getViewZ(const in float depth) {
			#if PERSPECTIVE_CAMERA == 1
				return perspectiveDepthToViewZ(depth, cameraNear, cameraFar);
			#else
				return orthographicDepthToViewZ(depth, cameraNear, cameraFar);
			#endif
		}

		vec3 computeNormalFromDepth(const vec2 uv) {
			vec2 size = vec2(textureSize(tDepth, 0));
			ivec2 p = ivec2(uv * size);
			float c0 = fetchDepth(p);
			float l2 = fetchDepth(p - ivec2(2, 0));
			float l1 = fetchDepth(p - ivec2(1, 0));
			float r1 = fetchDepth(p + ivec2(1, 0));
			float r2 = fetchDepth(p + ivec2(2, 0));
			float b2 = fetchDepth(p - ivec2(0, 2));
			float b1 = fetchDepth(p - ivec2(0, 1));
			float t1 = fetchDepth(p + ivec2(0, 1));
			float t2 = fetchDepth(p + ivec2(0, 2));
			float dl = abs((2.0 * l1 - l2) - c0);
			float dr = abs((2.0 * r1 - r2) - c0);
			float db = abs((2.0 * b1 - b2) - c0);
			float dt = abs((2.0 * t1 - t2) - c0);
			vec3 ce = getViewPosition(uv, c0).xyz;
			vec3 dpdx = (dl < dr) ?  ce - getViewPosition((uv - vec2(1.0 / size.x, 0.0)), l1).xyz
								: -ce + getViewPosition((uv + vec2(1.0 / size.x, 0.0)), r1).xyz;
			vec3 dpdy = (db < dt) ?  ce - getViewPosition((uv - vec2(0.0, 1.0 / size.y)), b1).xyz
								: -ce + getViewPosition((uv + vec2(0.0, 1.0 / size.y)), t1).xyz;
			return normalize(cross(dpdx, dpdy));
		}

		vec3 getViewNormal(const vec2 uv) {
			#if NORMAL_VECTOR_TYPE == 2
				return normalize(textureLod(tNormal, uv, 0.).rgb);
			#elif NORMAL_VECTOR_TYPE == 1
				return unpackRGBToNormal(textureLod(tNormal, uv, 0.).rgb);
			#else
				return computeNormalFromDepth(uv);
			#endif
		}

		vec3 getSceneUvAndDepth(vec3 sampleViewPos) {
			vec4 sampleClipPos = cameraProjectionMatrix * vec4(sampleViewPos, 1.);
			vec2 sampleUv = sampleClipPos.xy / sampleClipPos.w * 0.5 + 0.5;
			float sampleSceneDepth = getDepth(sampleUv);
			return vec3(sampleUv, sampleSceneDepth);
		}

		float sinusToPlane(vec3 pointOnPlane, vec3 planeNormal, vec3 point) {
			vec3 delta = point - pointOnPlane;
			float sinV = dot(planeNormal, normalize(delta));
			return sinV;
		}

		float getFallOff(float delta, float falloffDistance) {
			#if DISTANCE_FALL_OFF == 1
				float fallOff = smoothstep(0., 1., 1. - abs(delta) / falloffDistance);
			#else
				float fallOff = step(abs(delta), falloffDistance);
			#endif
			return fallOff;
		}
		
		void main() {
			float depth = getDepth(vUv.xy);
			if (depth == 1.0) {
				discard;
				return;
			}
			vec3 viewPos = getViewPosition(vUv, depth);
			vec3 viewNormal = getViewNormal(vUv);
			
			vec2 noiseResolution = vec2(textureSize(tNoise, 0));
			vec2 noiseUv = vUv * resolution / noiseResolution;
			vec4 noiseTexel = textureLod(tNoise, noiseUv, 0.0);
			vec3 randomVec = noiseTexel.xyz * 2.0 - 1.0;

			#if NV_ALIGNED_SAMPLES == 1
				vec3 tangent = normalize(randomVec - viewNormal * dot(randomVec, viewNormal));
				vec3 bitangent = cross(viewNormal, tangent);
				mat3 kernelMatrix = mat3(tangent, bitangent, viewNormal);
			#else
				vec3 tangent = normalize(vec3(randomVec.xy, 0.));
				vec3 bitangent = vec3(-tangent.y, tangent.x, 0.);
				mat3 kernelMatrix = mat3(tangent, bitangent, vec3(0., 0., 1.));
			#endif

			float radiusToUse = radius;
			float distanceFalloffToUse = thickness;
			#if SCREEN_SPACE_RADIUS == 1
				float radiusScale = getViewPosition(vec2(0.5 + float(SCREEN_SPACE_RADIUS_SCALE) / resolution.x, 0.0), depth).x;
				radiusToUse *= radiusScale;
				distanceFalloffToUse *= radiusScale;
			#endif

			const int DIRECTIONS = SAMPLES < 30 ? 3 : 5;
			const int STEPS = (SAMPLES + DIRECTIONS - 1) / DIRECTIONS;
		
			float ao = 0.0, totalWeight = 0.0;
			vec3 bentNormal = vec3(0.0);
			for (int i = 0; i < DIRECTIONS; ++i) {

				float angle = float(i) / float(DIRECTIONS) * PI;
				vec4 sampleDir = vec4(cos(angle), sin(angle), 0., 0.5 + 0.5 * noiseTexel.w); 
				sampleDir.xyz = normalize(kernelMatrix * sampleDir.xyz);

				vec3 viewDir = normalize(-viewPos.xyz);
				vec3 sliceBitangent = normalize(cross(sampleDir.xyz, viewDir));
				vec3 sliceTangent = cross(sliceBitangent, viewDir);
				vec3 normalInSlice = normalize(viewNormal - sliceBitangent * dot(viewNormal, sliceBitangent));
				
				vec3 tangentToNormalInSlice = cross(normalInSlice, sliceBitangent);
				vec2 cosHorizons = vec2(dot(viewDir, tangentToNormalInSlice), dot(viewDir, -tangentToNormalInSlice));
				for (int j = 0; j < STEPS; ++j) {
					vec3 sampleViewOffset = sampleDir.xyz * radiusToUse * sampleDir.w * pow(float(j + 1) / float(STEPS), distanceExponent);
					vec3 sampleViewPos = viewPos + sampleViewOffset;
			
					vec3 sampleSceneUvDepth = getSceneUvAndDepth(sampleViewPos);
					vec3 sampleSceneViewPos = getViewPosition(sampleSceneUvDepth.xy, sampleSceneUvDepth.z);
					float sceneSampleDist = abs(sampleSceneViewPos.z);
					float sampleDist = abs(sampleViewPos.z);	
					vec3 viewDelta = sampleSceneViewPos - viewPos;
					if (abs(viewDelta.z) < thickness) {
						float sampleCosHorizon = dot(viewDir, normalize(viewDelta));
						cosHorizons.x = max(cosHorizons.x, sampleCosHorizon);	
					}		
					sampleSceneUvDepth = getSceneUvAndDepth(viewPos - sampleViewOffset);
					sampleSceneViewPos = getViewPosition(sampleSceneUvDepth.xy, sampleSceneUvDepth.z);
					viewDelta = sampleSceneViewPos - viewPos;
					if (abs(viewDelta.z) < thickness) {
						float sampleCosHorizon = dot(viewDir, normalize(viewDelta));
						cosHorizons.y = max(cosHorizons.y, sampleCosHorizon);	
					}
				}

				vec2 sinHorizons = sqrt(1. - cosHorizons * cosHorizons);
				float nx = dot(normalInSlice, sliceTangent);
				float ny = dot(normalInSlice, viewDir);
				float nxb = 1. / 2. * (acos(cosHorizons.y) - acos(cosHorizons.x) + sinHorizons.x * cosHorizons.x - sinHorizons.y * cosHorizons.y);
				float nyb = 1. / 2. * (2. - cosHorizons.x * cosHorizons.x - cosHorizons.y * cosHorizons.y);
				float occlusion = nx * nxb + ny * nyb;
				ao += occlusion;
				bentNormal += nxb * sliceTangent + nyb * viewDir;
			}
			
			ao /= float(DIRECTIONS);	
			ao = pow(ao, scale);
			bentNormal = normalize(bentNormal);
			float alpha = acos(1. - ao);
			float f0 = alpha * (1. + pow(1. - alpha, 0.75) / 2.);

			//gl_FragColor = FRAGMENT_OUTPUT;

			vec4 diffuseColor = texture2D(tDiffuse, vUv);
			gl_FragColor = vec4(vec3(ao * 0.5) + vec3(ao * 0.5) * diffuseColor.rgb, 1.);
			
			//gl_FragColor = vec4(abs(bentNormal), 1.);
			//gl_FragColor = vec4(abs(viewNormal), 1.);
			//gl_FragColor = vec4(mix(abs(viewNormal), abs(bentNormal), step(0.5, vUv.x)), 1.);
			//gl_FragColor = vec4(vec3(f0), 1.);
		}`

};

const HBILDepthShader = {

	name: 'HBILDepthShader',

	defines: {
		'PERSPECTIVE_CAMERA': 1
	},

	uniforms: {

		'tDepth': { value: null },
		'cameraNear': { value: null },
		'cameraFar': { value: null },

	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`

		uniform sampler2D tDepth;

		uniform float cameraNear;
		uniform float cameraFar;

		varying vec2 vUv;

		#include <packing>

		float getLinearDepth( const in vec2 screenPosition ) {

			#if PERSPECTIVE_CAMERA == 1

				float fragCoordZ = texture2D( tDepth, screenPosition ).x;
				float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
				return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );

			#else

				return texture2D( tDepth, screenPosition ).x;

			#endif

		}

		void main() {

			float depth = getLinearDepth( vUv );
			gl_FragColor = vec4( vec3( 1.0 - depth ), 1.0 );

		}`

};

function generateAoSampleKernelInitializer( samples ) {

	const poissonDisk = generateAoSamples( samples );

	let glslCode = 'vec4[SAMPLES](';

	for ( let i = 0; i < samples; i ++ ) {

		const sample = poissonDisk[ i ];
		glslCode += `vec4(${sample.x}, ${sample.y}, ${sample.z}, ${sample.w})`;

		if ( i < samples - 1 ) {

			glslCode += ',';

		}

	}

	glslCode += ')';

	return glslCode;

}

function generateAoSamples( samples, cosineWeighted = false ) {

	// https://github.com/Rabbid76/research-sampling-hemisphere
	const kernel = [];
	for ( let sampleIndex = 0; sampleIndex < samples; sampleIndex ++ ) {

		const spiralAngle = sampleIndex * Math.PI * ( 3 - Math.sqrt( 5 ) );
		let z = 0.01 + ( sampleIndex / ( samples - 1 ) ) * 0.99;
		if ( cosineWeighted ) {

			z = Math.sqrt( z );

		}

		const radius = 1 - z;
		const x = Math.cos( spiralAngle ) * radius;
		const y = Math.sin( spiralAngle ) * radius;
		const scaleStep = 4;
		const scaleRange = Math.floor( samples / scaleStep );
		const scaleIndex = Math.floor( sampleIndex / scaleStep ) + ( sampleIndex % scaleStep ) * scaleRange;
		let scale = 1 - scaleIndex / samples;
		scale = 0.1 + 0.9 * scale;
		kernel.push( new Vector4( x, y, z, scale ) );

	}

	return kernel;

}

function generateMagicSquareNoise( size = 5 ) {

	const noiseSize = Math.floor( size ) % 2 === 0 ? Math.floor( size ) + 1 : Math.floor( size );
	const magicSquare = generateMagicSquare( noiseSize );
	const noiseSquareSize = magicSquare.length;
	const data = new Uint8Array( noiseSquareSize * 4 );
	for ( let inx = 0; inx < noiseSquareSize; ++ inx ) {

		const iAng = magicSquare[ inx ];
		const angle = ( 2 * Math.PI * iAng ) / noiseSquareSize;
		const randomVec = new Vector3(
			Math.cos( angle ),
			Math.sin( angle ),
			0
		).normalize();
		data[ inx * 4 ] = ( randomVec.x * 0.5 + 0.5 ) * 255;
		data[ inx * 4 + 1 ] = ( randomVec.y * 0.5 + 0.5 ) * 255;
		data[ inx * 4 + 2 ] = 127;
		data[ inx * 4 + 3 ] = 0;

	}

	const noiseTexture = new DataTexture( data, noiseSize, noiseSize );
	noiseTexture.wrapS = RepeatWrapping;
	noiseTexture.wrapT = RepeatWrapping;
	noiseTexture.needsUpdate = true;
	return noiseTexture;

}

function generateMagicSquare( size ) {

	const noiseSize =
	  Math.floor( size ) % 2 === 0 ? Math.floor( size ) + 1 : Math.floor( size );
	const noiseSquareSize = noiseSize * noiseSize;
	const magicSquare = Array( noiseSquareSize ).fill( 0 );
	let i = Math.floor( noiseSize / 2 );
	let j = noiseSize - 1;
	for ( let num = 1; num <= noiseSquareSize; ) {

	  if ( i === - 1 && j === noiseSize ) {

			j = noiseSize - 2;
			i = 0;

		} else {

			if ( j === noiseSize ) {

		  j = 0;

			}

			if ( i < 0 ) {

		  i = noiseSize - 1;

			}

		}

	  if ( magicSquare[ i * noiseSize + j ] !== 0 ) {

			j -= 2;
			i ++;
			continue;

		} else {

			magicSquare[ i * noiseSize + j ] = num ++;

		}

	  j ++;
	  i --;

	}

	return magicSquare;

}


export { generateAoSampleKernelInitializer, generateMagicSquareNoise, HBILShader, HBILDepthShader };
