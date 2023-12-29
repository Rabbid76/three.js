import {
	AddEquation,
	Color,
	CustomBlending,
	DataTexture,
	DepthStencilFormat,
	DepthTexture,
	DstAlphaFactor,
	DstColorFactor,
	HalfFloatType,
	MeshNormalMaterial,
	NearestFilter,
	NoBlending,
	RepeatWrapping,
	RGBAFormat,
	ShaderMaterial,
	UniformsUtils,
	UnsignedByteType,
	UnsignedInt248Type,
	WebGLRenderTarget,
	ZeroFactor
} from 'three';
import { Pass, FullScreenQuad } from './Pass.js';
import { generateAoSampleKernelInitializer, generateMagicSquareNoise, HBILShader, HBILDepthShader } from '../shaders/HBILShader.js';
import { generatePdSamplePointInitializer, PoissonDenoiseShader } from '../shaders/PoissonDenoiseShader.js';
import { CopyShader } from '../shaders/CopyShader.js';
import { SimplexNoise } from '../math/SimplexNoise.js';

class HBILPass extends Pass {

	constructor( scene, camera, width, height, parameters, aoParameters, pdParameters ) {

		super();

		this.width = ( width !== undefined ) ? width : 512;
		this.height = ( height !== undefined ) ? height : 512;
		this.clear = true;
		this.camera = camera;
		this.scene = scene;
		this.output = 0;
		this._renderGBuffer = true;
		this._visibilityCache = new Map();

		this.pdRings = 2.;
		this.pdRadiusExponent = 2.;
		this.pdSamples = 16;

		this.aoNoiseTextureMagicSquare = generateMagicSquareNoise();
		this.aoNoiseTextureRandom = this.generateNoise();
		this.pdNoiseTexture = this.generateNoise();

		this.hbilRenderTarget = new WebGLRenderTarget( this.width, this.height, { type: HalfFloatType } );
		this.pdRenderTarget = this.hbilRenderTarget.clone();

		this.hbilMaterial = new ShaderMaterial( {
			defines: Object.assign( {}, HBILShader.defines ),
			uniforms: UniformsUtils.clone( HBILShader.uniforms ),
			vertexShader: HBILShader.vertexShader,
			fragmentShader: HBILShader.fragmentShader,
			blending: NoBlending,
			depthTest: false,
			depthWrite: false,
		} );
		this.hbilMaterial.definesPERSPECTIVE_CAMERA = this.camera.isPerspectiveCamera ? 1 : 0;
		this.hbilMaterial.uniforms.tNoise.value = this.aoNoiseTextureMagicSquare;
		this.hbilMaterial.uniforms.resolution.value.set( this.width, this.height );
		this.hbilMaterial.uniforms.cameraNear.value = this.camera.near;
		this.hbilMaterial.uniforms.cameraFar.value = this.camera.far;

		this.normalMaterial = new MeshNormalMaterial();
		this.normalMaterial.blending = NoBlending;

		this.pdMaterial = new ShaderMaterial( {
			defines: Object.assign( {}, PoissonDenoiseShader.defines ),
			uniforms: UniformsUtils.clone( PoissonDenoiseShader.uniforms ),
			vertexShader: PoissonDenoiseShader.vertexShader,
			fragmentShader: PoissonDenoiseShader.fragmentShader,
			depthTest: false,
			depthWrite: false,
		} );
		this.pdMaterial.uniforms.tDiffuse.value = this.hbilRenderTarget.texture;
		this.pdMaterial.uniforms.tNoise.value = this.pdNoiseTexture;
		this.pdMaterial.uniforms.resolution.value.set( this.width, this.height );
		this.pdMaterial.uniforms.lumaPhi.value = 10;
		this.pdMaterial.uniforms.depthPhi.value = 2;
		this.pdMaterial.uniforms.normalPhi.value = 3;
		this.pdMaterial.uniforms.radius.value = 4;

		this.depthRenderMaterial = new ShaderMaterial( {
			defines: Object.assign( {}, HBILDepthShader.defines ),
			uniforms: UniformsUtils.clone( HBILDepthShader.uniforms ),
			vertexShader: HBILDepthShader.vertexShader,
			fragmentShader: HBILDepthShader.fragmentShader,
			blending: NoBlending
		} );
		this.depthRenderMaterial.uniforms.cameraNear.value = this.camera.near;
		this.depthRenderMaterial.uniforms.cameraFar.value = this.camera.far;

		this.copyMaterial = new ShaderMaterial( {
			uniforms: UniformsUtils.clone( CopyShader.uniforms ),
			vertexShader: CopyShader.vertexShader,
			fragmentShader: CopyShader.fragmentShader,
			transparent: true,
			depthTest: false,
			depthWrite: false,
			blendSrc: DstColorFactor,
			blendDst: ZeroFactor,
			blendEquation: AddEquation,
			blendSrcAlpha: DstAlphaFactor,
			blendDstAlpha: ZeroFactor,
			blendEquationAlpha: AddEquation
		} );

		this.fsQuad = new FullScreenQuad( null );

		this.originalClearColor = new Color();

		this.setGBuffer( parameters ? parameters.depthTexture : undefined, parameters ? parameters.normalTexture : undefined );

		if ( aoParameters !== undefined ) {

			this.updateHbilMaterial( aoParameters );

		}

		if ( pdParameters !== undefined ) {

			this.updatePdMaterial( pdParameters );

		}

	}

	dispose() {

		this.aoNoiseTextureMagicSquare.dispose();
		this.aoNoiseTextureRandom.dispose();
		this.pdNoiseTexture.dispose();
		this.normalRenderTarget.dispose();
		this.hbilRenderTarget.dispose();
		this.pdRenderTarget.dispose();
		this.normalMaterial.dispose();
		this.pdMaterial.dispose();
		this.copyMaterial.dispose();
		this.depthRenderMaterial.dispose();
		this.fsQuad.dispose();

	}

	setGBuffer( depthTexture, normalTexture ) {

		if ( depthTexture !== undefined ) {

			this.depthTexture = depthTexture;
			this.normalTexture = normalTexture;
			this._renderGBuffer = false;

		} else {

			this.depthTexture = new DepthTexture();
			this.depthTexture.format = DepthStencilFormat;
			this.depthTexture.type = UnsignedInt248Type;

			this.normalRenderTarget = new WebGLRenderTarget( this.width, this.height, {
				minFilter: NearestFilter,
				magFilter: NearestFilter,
				type: HalfFloatType,
				depthTexture: this.depthTexture
			} );
			this.normalTexture = this.normalRenderTarget.texture;
			this._renderGBuffer = true;

		}

		const normalVectorType = ( this.normalTexture ) ? 1 : 0;
		const depthValueSource = ( this.depthTexture === this.normalTexture ) ? 'w' : 'x';

		this.hbilMaterial.defines.NORMAL_VECTOR_TYPE = normalVectorType;
		this.hbilMaterial.defines.DEPTH_SWIZZLING = depthValueSource;
		this.hbilMaterial.uniforms.tNormal.value = this.normalTexture;
		this.hbilMaterial.uniforms.tDepth.value = this.depthTexture;

		this.pdMaterial.defines.NORMAL_VECTOR_TYPE = normalVectorType;
		this.pdMaterial.defines.DEPTH_SWIZZLING = depthValueSource;
		this.pdMaterial.uniforms.tNormal.value = this.normalTexture;
		this.pdMaterial.uniforms.tDepth.value = this.depthTexture;

		this.depthRenderMaterial.uniforms.tDepth.value = this.normalRenderTarget.depthTexture;

	}

	updateHbilMaterial( parameters ) {

		if ( parameters.radius !== undefined ) {

			this.hbilMaterial.uniforms.radius.value = parameters.radius;

		}

		if ( parameters.distanceExponent !== undefined ) {

			this.hbilMaterial.uniforms.distanceExponent.value = parameters.distanceExponent;

		}

		if ( parameters.thickness !== undefined ) {

			this.hbilMaterial.uniforms.thickness.value = parameters.thickness;

		}

		if ( parameters.bias !== undefined ) {

			this.hbilMaterial.uniforms.bias.value = parameters.bias;

		}

		if ( parameters.scale !== undefined ) {

			this.hbilMaterial.uniforms.scale.value = parameters.scale;

		}

		if ( parameters.samples !== undefined && parameters.samples !== this.hbilMaterial.defines.SAMPLES ) {

			this.hbilMaterial.defines.SAMPLES = parameters.samples;
			this.hbilMaterial.defines.SAMPLE_VECTORS = generateAoSampleKernelInitializer( parameters.samples );
			this.hbilMaterial.needsUpdate = true;

		}

		if ( parameters.distanceFallOff !== undefined && ( parameters.distanceFallOff ? 1 : 0 ) !== this.hbilMaterial.defines.DISTANCE_FALL_OFF ) {

			this.hbilMaterial.defines.DISTANCE_FALL_OFF = parameters.distanceFallOff ? 1 : 0;
			this.hbilMaterial.needsUpdate = true;

		}

		if ( parameters.clipRangeCheck !== undefined && ( parameters.clipRangeCheck ? 1 : 0 ) !== this.hbilMaterial.defines.CLIP_RANGE_CHECK ) {

			this.hbilMaterial.defines.CLIP_RANGE_CHECK = parameters.clipRangeCheck ? 1 : 0;
			this.hbilMaterial.needsUpdate = true;

		}

		if ( parameters.depthRelativeBias !== undefined && ( parameters.depthRelativeBias ? 1 : 0 ) !== this.hbilMaterial.defines.BIAS_RELATIVE_TO_DEPTH ) {

			this.hbilMaterial.defines.BIAS_RELATIVE_TO_DEPTH = parameters.depthRelativeBias ? 1 : 0;
			this.hbilMaterial.needsUpdate = true;

		}

		if ( parameters.nvAlignedSamples !== undefined && ( parameters.nvAlignedSamples ? 1 : 0 ) !== this.hbilMaterial.defines.NORMAL_VECTOR_ALIGNED_SAMPLES ) {

			this.hbilMaterial.defines.NV_ALIGNED_SAMPLES = parameters.nvAlignedSamples ? 1 : 0;
			this.hbilMaterial.needsUpdate = true;

		}

		if ( parameters.screenSpaceRadius !== undefined && ( parameters.screenSpaceRadius ? 1 : 0 ) !== this.hbilMaterial.defines.SCREEN_SPACE_RADIUS ) {

			this.hbilMaterial.defines.SCREEN_SPACE_RADIUS = parameters.screenSpaceRadius ? 1 : 0;
			this.hbilMaterial.needsUpdate = true;

		}

		if ( parameters.aoNoiseType !== undefined ) {

			if ( parameters.aoNoiseType === 'magic-square' ) {

				this.hbilMaterial.uniforms.tNoise.value = this.aoNoiseTextureMagicSquare;

			} else if ( parameters.aoNoiseType === 'random' ) {

				this.hbilMaterial.uniforms.tNoise.value = this.aoNoiseTextureRandom;

			}

		}

	}

	updatePdMaterial( parameters ) {

		let updateShader = false;

		if ( parameters.lumaPhi !== undefined ) {

			this.pdMaterial.uniforms.lumaPhi.value = parameters.lumaPhi;

		}

		if ( parameters.depthPhi !== undefined ) {

			this.pdMaterial.uniforms.depthPhi.value = parameters.depthPhi;

		}

		if ( parameters.normalPhi !== undefined ) {

			this.pdMaterial.uniforms.normalPhi.value = parameters.normalPhi;

		}

		if ( parameters.radius !== undefined && parameters.radius !== this.radius ) {

			this.pdMaterial.uniforms.radius.value = parameters.radius;

		}

		if ( parameters.radiusExponent !== undefined && parameters.radiusExponent !== this.pdRadiusExponent ) {

			this.pdRadiusExponent = parameters.radiusExponent;
			updateShader = true;

		}

		if ( parameters.rings !== undefined && parameters.rings !== this.pdRings ) {

			this.pdRings = parameters.rings;
			updateShader = true;

		}

		if ( parameters.samples !== undefined && parameters.samples !== this.pdSamples ) {

			this.pdSamples = parameters.samples;
			updateShader = true;

		}

		if ( updateShader ) {

			this.pdMaterial.defines.SAMPLES = this.pdSamples;
			this.pdMaterial.defines.SAMPLE_VECTORS = generatePdSamplePointInitializer( this.pdSamples, this.pdRings, this.pdRadiusExponent );
			this.pdMaterial.needsUpdate = true;

		}

	}

	render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {

		// render normals and depth (honor only meshes, points and lines do not contribute to AO)

		if ( this._renderGBuffer ) {

			this.overrideVisibility();
			this.renderOverride( renderer, this.normalMaterial, this.normalRenderTarget, 0x7777ff, 1.0 );
			this.restoreVisibility();

		}

		// render AO

		this.hbilMaterial.uniforms.cameraNear.value = this.camera.near;
		this.hbilMaterial.uniforms.cameraFar.value = this.camera.far;
		this.hbilMaterial.uniforms.cameraProjectionMatrix.value.copy( this.camera.projectionMatrix );
		this.hbilMaterial.uniforms.cameraProjectionMatrixInverse.value.copy( this.camera.projectionMatrixInverse );
		this.hbilMaterial.uniforms.tDiffuse.value = readBuffer.texture;
		this.renderPass( renderer, this.hbilMaterial, this.hbilRenderTarget, 0xffffff, 1.0 );

		// render poisson denoise

		this.pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy( this.camera.projectionMatrixInverse );
		this.renderPass( renderer, this.pdMaterial, this.pdRenderTarget, 0xffffff, 1.0 );

		// output result to screen

		switch ( this.output ) {

			case HBILPass.OUTPUT.Diffuse:

				this.copyMaterial.uniforms.tDiffuse.value = readBuffer.texture;
				this.copyMaterial.blending = NoBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : writeBuffer );

				break;

			case HBILPass.OUTPUT.IL:

				this.copyMaterial.uniforms.tDiffuse.value = this.hbilRenderTarget.texture;
				this.copyMaterial.blending = NoBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : writeBuffer );

				break;

			case HBILPass.OUTPUT.Denoise:

				this.copyMaterial.uniforms.tDiffuse.value = this.pdRenderTarget.texture;
				this.copyMaterial.blending = NoBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : writeBuffer );

				break;

			case HBILPass.OUTPUT.Depth:

				this.depthRenderMaterial.uniforms.cameraNear.value = this.camera.near;
				this.depthRenderMaterial.uniforms.cameraFar.value = this.camera.far;
				this.renderPass( renderer, this.depthRenderMaterial, this.renderToScreen ? null : writeBuffer );

				break;

			case HBILPass.OUTPUT.Normal:

				this.copyMaterial.uniforms.tDiffuse.value = this.normalRenderTarget.texture;
				this.copyMaterial.blending = NoBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : writeBuffer );

				break;

			case HBILPass.OUTPUT.Default:

				this.copyMaterial.uniforms.tDiffuse.value = readBuffer.texture;
				this.copyMaterial.blending = NoBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : writeBuffer );

				this.copyMaterial.uniforms.tDiffuse.value = this.pdRenderTarget.texture;
				this.copyMaterial.blending = CustomBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : writeBuffer );

				break;

			default:
				console.warn( 'THREE.HBILPass: Unknown output type.' );

		}

	}

	renderPass( renderer, passMaterial, renderTarget, clearColor, clearAlpha ) {

		// save original state
		renderer.getClearColor( this.originalClearColor );
		const originalClearAlpha = renderer.getClearAlpha();
		const originalAutoClear = renderer.autoClear;

		renderer.setRenderTarget( renderTarget );

		// setup pass state
		renderer.autoClear = false;
		if ( ( clearColor !== undefined ) && ( clearColor !== null ) ) {

			renderer.setClearColor( clearColor );
			renderer.setClearAlpha( clearAlpha || 0.0 );
			renderer.clear();

		}

		this.fsQuad.material = passMaterial;
		this.fsQuad.render( renderer );

		// restore original state
		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( this.originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );

	}

	renderOverride( renderer, overrideMaterial, renderTarget, clearColor, clearAlpha ) {

		renderer.getClearColor( this.originalClearColor );
		const originalClearAlpha = renderer.getClearAlpha();
		const originalAutoClear = renderer.autoClear;

		renderer.setRenderTarget( renderTarget );
		renderer.autoClear = false;

		clearColor = overrideMaterial.clearColor || clearColor;
		clearAlpha = overrideMaterial.clearAlpha || clearAlpha;

		if ( ( clearColor !== undefined ) && ( clearColor !== null ) ) {

			renderer.setClearColor( clearColor );
			renderer.setClearAlpha( clearAlpha || 0.0 );
			renderer.clear();

		}

		this.scene.overrideMaterial = overrideMaterial;
		renderer.render( this.scene, this.camera );
		this.scene.overrideMaterial = null;

		// restore original state

		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( this.originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );

	}

	setSize( width, height ) {

		this.width = width;
		this.height = height;

		this.hbilRenderTarget.setSize( width, height );
		this.normalRenderTarget.setSize( width, height );
		this.pdRenderTarget.setSize( width, height );

		this.hbilMaterial.uniforms.resolution.value.set( width, height );
		this.hbilMaterial.uniforms.cameraProjectionMatrix.value.copy( this.camera.projectionMatrix );
		this.hbilMaterial.uniforms.cameraProjectionMatrixInverse.value.copy( this.camera.projectionMatrixInverse );

		this.pdMaterial.uniforms.resolution.value.set( width, height );
		this.pdMaterial.uniforms.cameraProjectionMatrixInverse.value.copy( this.camera.projectionMatrixInverse );

	}

	overrideVisibility() {

		const scene = this.scene;
		const cache = this._visibilityCache;

		scene.traverse( function ( object ) {

			cache.set( object, object.visible );

			if ( object.isPoints || object.isLine ) object.visible = false;

		} );

	}

	restoreVisibility() {

		const scene = this.scene;
		const cache = this._visibilityCache;

		scene.traverse( function ( object ) {

			const visible = cache.get( object );
			object.visible = visible;

		} );

		cache.clear();

	}

	generateNoise( size = 64 ) {

		const simplex = new SimplexNoise();

		const arraySize = size * size * 4;
		const data = new Uint8Array( arraySize );

		for ( let i = 0; i < size; i ++ ) {

			for ( let j = 0; j < size; j ++ ) {

				const x = i;
				const y = j;

				data[ ( i * size + j ) * 4 ] = ( simplex.noise( x, y ) * 0.5 + 0.5 ) * 255;
				data[ ( i * size + j ) * 4 + 1 ] = ( simplex.noise( x + size, y ) * 0.5 + 0.5 ) * 255;
				data[ ( i * size + j ) * 4 + 2 ] = ( simplex.noise( x, y + size ) * 0.5 + 0.5 ) * 255;
				data[ ( i * size + j ) * 4 + 3 ] = ( simplex.noise( x + size, y + size ) * 0.5 + 0.5 ) * 255;

			}

		}

		const noiseTexture = new DataTexture( data, size, size, RGBAFormat, UnsignedByteType );
		noiseTexture.wrapS = RepeatWrapping;
		noiseTexture.wrapT = RepeatWrapping;
		noiseTexture.needsUpdate = true;

		return noiseTexture;

	}

}

HBILPass.OUTPUT = {
	'Default': 0,
	'Diffuse': 1,
	'Depth': 2,
	'Normal': 3,
	'IL': 4,
	'Denoise': 5,
};

export { HBILPass };
