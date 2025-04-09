try:
    import py21cmfast as p21c
except:
    raise ImportError('To use the density and halo catalogues from 21cmfast, install the package from https://github.com/21cmfast/21cmFAST/')

import time
import tools21cm as t2c
import logging
logger = logging.getLogger(__name__)

from .cosmo import *
from .functions import *
from .structs.parameters import Parameters
from .io import Handler
"""
def prepare():
    user_params = p21c.UserParams({"HII_DIM": parameters.simulation.Ncell, "DIM": parameters.simulation.Ncell * 3,
                                   "BOX_LEN": parameters.simulation.Lbox / parameters.cosmo.h,
                                   "USE_INTERPOLATION_TABLES": True,
                                   # "FIXED_IC": True,
                                   "N_THREADS": parameters.simulation.cores,
                                   })

    cosmo_params = p21c.CosmoParams(SIGMA_8=parameters.cosmology.sigma_8,
                                    hlittle=parameters.cosmology.h,
                                    OMm=parameters.cosmology.Om,
                                    OMb=parameters.cosmology.Ob,
                                    POWER_INDEX=parameters.cosmology.ns,
                                    )
    random_seed = parameters.simulation.random_seed


    pslin, klin = t2c.power_spectrum_1d(IC.hires_density, kbins=20, box_dims=user_params.BOX_LEN)
    return IC, pslin, klin


def simulate_matter_21cmfast(parameters: Parameters, redshift_list=None, IC=None, data_dir=None):
    
    start_time = time.time()
    print('Simulating matter evolution with 21cmFast...')

    user_params = p21c.UserParams({"HII_DIM": parameters.simulation.Ncell, "DIM": parameters.simulation.Ncell * 3,
                                   "BOX_LEN": parameters.simulation.Lbox / parameters.cosmology.h,
                                   "USE_INTERPOLATION_TABLES": True,
                                   # "FIXED_IC": True,
                                   "N_THREADS": parameters.simulation.cores,
                                   })

    cosmo_params = p21c.CosmoParams(SIGMA_8=parameters.cosmology.sigma8,
                                    hlittle=parameters.cosmology.h,
                                    OMm=parameters.cosmology.Om,
                                    OMb=parameters.cosmology.Ob,
                                    POWER_INDEX=parameters.cosmology.ns,
                                    )

    random_seed = parameters.simulation.random_seed
    print('random seed: ', random_seed)

    if redshift_list is None: 
        redshift_list = def_redshifts(parameters)

    if parameters.simulation.dens_field  is None:
        print('We strongly advice to specify a path for param.sim.dens_field to write the densities.')
        print('It should be of the form path+dir+name. For instance param.sim.dens_field = ./data_dir/dens_21cmFast_z')
    else:
        print('We will store the density fields in', parameters.simulation.dens_field)

    if parameters.simulation.halo_catalogs is None:
        print('We strongly advice to specify a path for param.sim.halo_catalogs to write the halo catalogues.')
        print('It should be of the form path+dir+name. For instance param.sim.halo_catalogs = ./dir_halos/halos_21cmFast_z')
    else:
        print('We will store halo catalogs in', parameters.simulation.halo_catalogs)
    
    dens_dict = {}
    halo_catalog_dict = {}
    with p21c.global_params.use(INITIAL_REDSHIFT=300, CLUMPING_FACTOR=2.0):
        for redshift in redshift_list:
            if IC is None:
                IC, pslin, klin = initialise_21cmfast(parameters)

            perturbed_field = p21c.perturb_field(
                redshift=redshift,
                init_boxes=IC,
                # user_params=user_params,
                # cosmo_params=cosmo_params,
                # astro_params=astro_params,
                # random_seed=random_seed,
                write=data_dir,
                direc=data_dir,
            )
            halo_list = p21c.perturb_halo_list(
                redshift=redshift,
                init_boxes=IC,
                # user_params=user_params,
                # cosmo_params=cosmo_params,
                # astro_params=astro_params,
                # random_seed=random_seed,
                write=data_dir,
                direc=data_dir,
            )

            h0 = parameters.cosmology.h
            Lbox = parameters.simulation.Lbox
            print('param.sim.Lbox is in Mpc/h. Halo catalogs catalogs have masses in Msol/h and positions in Mpc/h.')
            dens = perturbed_field.density
            halo_list = {'X': halo_list.halo_coords[:, 0] * Lbox / user_params.HII_DIM,
                         'Y': halo_list.halo_coords[:, 1] * Lbox / user_params.HII_DIM,
                         'Z': halo_list.halo_coords[:, 2] * Lbox / user_params.HII_DIM,
                         'M': halo_list.halo_masses * h0,
                         'z': redshift, 'Lbox': Lbox
                         }
            print('param.sim.Lbox is in Mpc/h. Halo catalogs catalogs have masses in Msol/h and positions in Mpc/h.')
            dens = perturbed_field.density
            halo_list = {'X': halo_list.halo_coords[:, 0] * Lbox / user_params.HII_DIM,
                         'Y': halo_list.halo_coords[:, 1] * Lbox / user_params.HII_DIM,
                         'Z': halo_list.halo_coords[:, 2] * Lbox / user_params.HII_DIM,
                         'M': halo_list.halo_masses * h0,
                         'z': redshift, 'Lbox': Lbox
                         }


            try:
                save_f(obj=dens, file=parameters.simulation.dens_field + z_string_format(redshift) + '.0')
                save_f(obj=halo_list, file=parameters.simulation.halo_catalogs + z_string_format(redshift))
            except:
                pass
            # dens_dict[f'{redshift:05.2f}'] = dens
            # halo_catalog_dict[f'{redshift:05.2f}'] = halo_list
            dens_dict[redshift] = dens
            halo_catalog_dict[redshift] = halo_list

    end_time = time.time()
    print('...done | Runtime =', print_time(end_time - start_time))
    return {'dens': dens_dict, 'halo_list': halo_catalog_dict}

"""
def generate_haloes_and_density(
    parameters: Parameters,
    handler: Handler = None,
):
    start_time = time.process_time()
    logger.info('Simulating matter evolution with 21cmFast')

    # Convert the parameters to the format required by py21cmfast
    # TODO don't regenerate if they exist already!
    user_params = p21c.UserParams({
        "HII_DIM": parameters.simulation.Ncell,
        "DIM": parameters.simulation.Ncell * 3,
        "BOX_LEN": parameters.simulation.Lbox / parameters.cosmology.h,
        "USE_INTERPOLATION_TABLES": True,
        # "FIXED_IC": True,
        "N_THREADS": parameters.simulation.cores,
    })

    cosmo_params = p21c.CosmoParams(
        SIGMA_8 = parameters.cosmology.sigma_8,
        hlittle = parameters.cosmology.h,
        OMm = parameters.cosmology.Om,
        OMb = parameters.cosmology.Ob,
        POWER_INDEX = parameters.cosmology.ns,
    )

    global_params = {
        "INITIAL_REDSHIFT": 300,
        "CLUMPING_FACTOR": 2.0,
    }

    halo_file_list = []
    dens_file_list = []

    Lbox = parameters.simulation.Lbox
    logger.debug(f'{parameters.simulation.Lbox=} in Mpc/h. Halo catalogs catalogs have masses in Msol/h and positions in Mpc/h.')

    # with p21c.global_params.use(**global_params):
    with p21c.global_params.use(INITIAL_REDSHIFT=300, CLUMPING_FACTOR=2.0):
        for redshift in parameters.solver.Nz:
            IC = p21c.initial_conditions(
                user_params = user_params,
                cosmo_params = cosmo_params,
                random_seed = parameters.simulation.random_seed,
            )

            # reuse the handler caching logic
            file_root = handler.file_root / f"21cmfast_{parameters.unique_hash()}"
            file_root.mkdir(parents=True, exist_ok=True)
            halo_fname = file_root / f'halos_21cmFast_z{redshift}'
            field_fname = file_root / f'dens_21cmFast_z{redshift}'
            
            if halo_fname.exists() and field_fname.exists():
                logger.debug(f"Skipping existing files: {halo_fname}, {field_fname}")

            else:
                logger.debug(f"Generating new files: {halo_fname}, {field_fname}")
                perturbed_field = p21c.perturb_field(
                    redshift = redshift,
                    # TODO directly pass it the redshift list
                    init_boxes = IC,
                    # user_params = user_params,
                    # cosmo_params = cosmo_params,
                    # astro_params = astro_params,
                    # random_seed = random_seed,
                )
                halo_list = p21c.perturb_halo_list(
                    redshift = redshift,
                    # TODO directly pass it the redshift list
                    init_boxes = IC,
                    # user_params = user_params,
                    # cosmo_params = cosmo_params,
                    # astro_params = astro_params,
                    # random_seed = random_seed,
                )

                halo_list.write(halo_fname)
                perturbed_field.write(field_fname)
            
            halo_file_list.append(halo_fname)
            dens_file_list.append(field_fname)


            # TODO - return slightly different format when reading
            # halo_list = {
            #     'X': halo_list.halo_coords[:, 0] * Lbox / user_params.HII_DIM,
            #     'Y': halo_list.halo_coords[:, 1] * Lbox / user_params.HII_DIM,
            #     'Z': halo_list.halo_coords[:, 2] * Lbox / user_params.HII_DIM,
            #     'M': halo_list.halo_masses * parameters.cosmology.h,
            #     'z': redshift,
            #     'Lbox': Lbox
            # }

    logger.info(f"Finished generating halos in {time.process_time() - start_time} seconds")
    return halo_file_list, dens_file_list
