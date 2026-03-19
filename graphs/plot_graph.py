import torch
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from anemoi.utils.config import DotDict
import matplotlib.patches as patches
import yaml
import sys

crs_cerra = ccrs.LambertConformal(
    central_longitude=8.0, 
    central_latitude=50, 
    standard_parallels=(50, 50), 
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    cutoff=-30
)

PROJECTIONS = {
        'CERRA' : crs_cerra,
        'PLATECARREE' : ccrs.PlateCarree(),
    }

cerra_600km_boundary_ticks = [-4*(10**6)+i*10**6 for i in range(9)]

def load_yaml(fn, **kwargs):
    with open(fn, 'r') as f:
        d = yaml.safe_load(f)
    d['projection'] = PROJECTIONS[d.get('projection', 'PLATECARREE')]
    d.update(kwargs)
    return d


def get_plot_data(graph, extent = None):
    if isinstance(graph, str):
        graph = torch.load(graph, map_location=torch.device('cpu'), weights_only=False)
    data = np.rad2deg(graph['data'].x.numpy())
    hidden = np.rad2deg(graph['hidden'].x.numpy())
    data[:,1] = (data[:,1] + 180) % 360 - 180
    hidden[:,1] = (hidden[:,1] + 180) % 360 - 180
    lam_mask = graph['data']["cutout"].numpy()[:,0]
    lam = data[lam_mask]
    bnd = data[~lam_mask]
    edges = graph[('data','to','hidden')]["edge_index"]
    x={}
    y={}
    x['lam'] = lam[:,1]
    y['lam'] = lam[:,0]
    x['bnd'] = bnd[:,1]
    y['bnd'] = bnd[:,0]
    x['hid'] = hidden[:,1]
    y['hid'] = hidden[:,0]
    x['src'] = data[edges[0,:],1]
    y['src']= data[edges[0,:],0]
    x['tgt'] = hidden[edges[1,:],1]
    y['tgt']= hidden[edges[1,:],0]
    x=DotDict(x)
    y=DotDict(y)
    if extent:
        for typ in ['lam', 'bnd', 'hid']:
            x[typ], y[typ] = crop_nodes(x[typ], y[typ], *extent)
        x.src, y.src, x.tgt, y.tgt = crop_edges(x.src, y.src, x.tgt, y.tgt, *extent)
    return (x, y)

def get_extent_mask(x, y, y_max, x_min, y_min, x_max):
    x_max_mask = np.where(x < x_max , True, False)
    x_min_mask = np.where(x > x_min , True, False)
    y_max_mask = np.where(y < y_max , True, False)
    y_min_mask = np.where(y > y_min , True, False)
    x_mask = np.logical_and(x_max_mask, x_min_mask)
    y_mask = np.logical_and(y_max_mask, y_min_mask)
    return np.logical_and(x_mask, y_mask)

def crop_nodes(x, y, y_max, x_min, y_min, x_max):
    extent_mask = get_extent_mask(x,y, y_max, x_min, y_min, x_max)
    return (x[extent_mask], y[extent_mask])

def crop_edges(x_s, y_s, x_t, y_t, y_max, x_min, y_min, x_max):
    src_extent_mask = get_extent_mask(x_s, y_s, y_max, x_min, y_min, x_max)
    tgt_extent_mask = get_extent_mask(x_t, y_t, y_max, x_min, y_min, x_max)
    extent_mask = np.logical_or(src_extent_mask, tgt_extent_mask)
    return (x_s[extent_mask], y_s[extent_mask], x_t[extent_mask], y_t[extent_mask])

def transform_extent(extent, proj):
    latlon_extent = extent
    if extent:
        xs = np.array([extent[1],       extent[1],        extent[1], (extent[1]+extent[3])/2,  extent[3],        extent[3],         extent[3], (extent[1]+extent[3])/2 ])
        ys = np.array([extent[0],(extent[0]+extent[2])/2, extent[2],         extent[2],        extent[2],  (extent[2]+extent[0])/2, extent[0],          extent[0] ])
        pts = ccrs.PlateCarree().transform_points(proj, xs, ys).transpose()
        N = pts[1].max()
        W = pts[0].min()
        S = pts[1].min()
        E = pts[0].max()
        latlon_extent = [N, W, S, E]
    return latlon_extent

def plot_graph(graph, **kwargs):
    print(f"plotting {graph}")
    yaml_fn = kwargs.pop('yaml', "")
    if yaml_fn:
        kwargs = load_yaml(yaml_fn, **kwargs)
    proj = kwargs.pop('projection', ccrs.PlateCarree())
    sup_title_kwargs = kwargs.pop('sup_title', {'title':'plot of the day'})
    plot_info = kwargs.pop('plot_info',kwargs)
    if isinstance(plot_info,dict):
        plot_info = [plot_info]
    n_plots = len(plot_info)
    # set up plot in projection of choice
    fig , axs = plt.subplots(1, n_plots, figsize=(27,9*n_plots),subplot_kw={"projection": proj})
    for i in range(n_plots):
        ax =axs
        if n_plots > 1:
            ax = axs[i]
        prep_ax(ax, graph, proj, **plot_info[i])
    # plt.legend(bbox_to_anchor=(1.1, 1.05),fontsize=20)
    title = sup_title_kwargs.pop('title')
    fig.suptitle(title, **sup_title_kwargs)
    plt.tight_layout()
    fn = kwargs.get('file_name', "")
    if fn:
        plt.savefig(fn, dpi=200)
    else:
        plt.show()

def prep_ax(ax, graph, proj, **kwargs):
    extent = kwargs.pop('extent', None)
    borders = kwargs.pop('borders', False)
    latlon_extent = transform_extent(extent, proj)
    zoom_box_kwargs = kwargs.pop('zoom_box', None)
    title_kwargs = kwargs.pop('sub_title', {'title':'just another subplot'})
    # get locations of all graph nodes and edge sources, targets, cropped to extent
    x, y = get_plot_data(graph, extent =  latlon_extent)
    
       
    # plot edges
    edge_info = kwargs.pop('EDGES', None)
    if edge_info:
        ax.plot([x.src,x.tgt],[y.src,y.tgt],transform=ccrs.PlateCarree(), zorder=10, **edge_info)
    # plot nodes
    d = {'ERA5':'bnd', 'CERRA':'lam', 'HIDDEN':'hid'}
    for label in list(kwargs):
        typ = d[label]
        ax.scatter(x[typ], y[typ],
                    transform=ccrs.PlateCarree(),
                    label=label, zorder = 20, 
                    **kwargs[label]
                    )
    # add some geography and other stuff 
    ax.coastlines(zorder=10)
    if extent:
        plt_extent = [extent[1], extent[3], extent[2], extent[0]]
        ax.set_extent(plt_extent,crs=proj)
    # ax.gridlines(draw_labels=False)
    ax.gridlines(crs=crs_cerra, draw_labels=False,
                linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    if borders:
        ax.add_feature(cfeature.BORDERS)
    #ax.set_xticks(cerra_600km_boundary_ticks)
    #ax.set_yticks(cerra_600km_boundary_ticks)
    title = title_kwargs.pop('title')
    ax.set_title(title, **title_kwargs)
    if zoom_box_kwargs:
        zoom_extent = zoom_box_kwargs.pop('extent')
        xy_zoom = [(zoom_extent[1], zoom_extent[0]),
                (zoom_extent[1], zoom_extent[2]),
                (zoom_extent[3], zoom_extent[2]),
                (zoom_extent[3], zoom_extent[0])
                ]
        rect = patches.Polygon(xy_zoom, **zoom_box_kwargs)
        ax.add_patch(rect)

def set_nested(d, keys, value):
    """Helper: assign value into nested dictionary based on list of keys."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value

def parse_value(value):
    """Try to cast value into int/float/bool if possible."""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # keep as string

def main():
    args = sys.argv[1:]  # skip script name
    kwargs = {}

    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            keys = key.split('.')  # support nested keys
            set_nested(kwargs, keys, parse_value(value))
        else:
            print(f"Warning: ignoring argument '{arg}' (no '=')")
    assert 'graph' in kwargs, "You need to specify a graph file via graph={graph_file}"
    graph = kwargs.pop('graph')
    plot_graph(graph, **kwargs)


if __name__ == "__main__":
    main()