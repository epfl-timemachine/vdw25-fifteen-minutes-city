# -*- coding: utf-8 -*-
"""
15-Minute Reachable Area Analysis for Venice (2024 networks)
-----------------------------------------------------------

本脚本读取 2024 年的街道 (步行) 与运河 (船行) GeoJSON 数据，
构建加权网络（以时间为权重），并为每栋建筑计算 15 分钟可达范围
(isochrones)。生成结果写入 `buildings_isochrones_15min.geojson`。

依赖库：
    geopandas, shapely, networkx, momepy, tqdm

使用方法：
    python 15_min_area_analysis.py

如需调整参数 (速度、时间阈值、缓冲距离等），请在脚本顶部修改常量。
"""
from pathlib import Path
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from shapely.ops import unary_union
import momepy as mm
from tqdm import tqdm
import pandas as pd

# ---------------------------
# 1. 路径与参数
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'venice-data-week-data' / '2024-geometries'
STREETS_FILE = DATA_DIR / '2024_Streets_EPSG32633.geojson'
CANALS_FILE = DATA_DIR / '2024_Canals_EPSG32633.geojson'
BUILDINGS_FILE = DATA_DIR / '2024_Edifici_EPSG32633.geojson'
OUTPUT_FILE = BASE_DIR / 'buildings_isochrones_15min.geojson'

# 行走与船速（米/分钟）
WALK_SPEED = 3600 / 60      # 5 km/h
BOAT_SPEED = 6000 / 60      # 6 km/h
TIME_MIN = 15               # 分钟阈值
BUFFER_M = 20               # 生成多边形时的缓冲半径 (米)


def build_graph(gdf: gpd.GeoDataFrame, speed_m_per_min: float) -> nx.Graph:
    """将线要素转换为 networkx 图，并以时间作为权重。"""
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf['length'] = gdf.geometry.length
    gdf['time'] = gdf['length'] / speed_m_per_min
    return mm.gdf_to_nx(gdf, approach='primal', length='time')


def snap_node_index(point, node_gdf, sindex) -> int:
    """返回距离 point 最近的节点编号。"""
    nearest_idx = list(sindex.nearest(point.bounds, 1))[0]
    return node_gdf.iloc[nearest_idx]['node']


def main():
    print('Loading layers...')
    streets = gpd.read_file(STREETS_FILE)
    canals = gpd.read_file(CANALS_FILE)
    buildings = gpd.read_file(BUILDINGS_FILE)

    # 确保 CRS 为 EPSG:32633
    for layer, name in zip([streets, canals, buildings], ['streets', 'canals', 'buildings']):
        if layer.crs.to_epsg() != 32633:
            raise ValueError(f'{name} CRS must be EPSG:32633')

    print('Building graphs...')
    G_walk = build_graph(streets, WALK_SPEED)
    G_boat = build_graph(canals, BOAT_SPEED)
    G = nx.compose(G_walk, G_boat)  # 合并两种交通方式

    # 创建节点 GeoDataFrame 便于最近邻查询
    node_ids, node_geoms = zip(*[(n, data['geometry']) for n, data in G.nodes(data=True)])
    node_gdf = gpd.GeoDataFrame({'node': node_ids}, geometry=list(node_geoms), crs=streets.crs)
    sindex = node_gdf.sindex

    print('Computing isochrones...')
    isochrones = []
    cutoff = TIME_MIN  # 单位：分钟

    for _, bld in tqdm(buildings.iterrows(), total=len(buildings)):
        centroid = bld.geometry.centroid
        source_node = snap_node_index(centroid, node_gdf, sindex)
        lengths = nx.single_source_dijkstra_path_length(G, source_node, cutoff=cutoff, weight='time')

        # 收集可达边
        reachable_edges = []
        for u, v, data in G.edges(data=True):
            if u in lengths or v in lengths:
                reachable_edges.append(LineString([G.nodes[u]['geometry'], G.nodes[v]['geometry']]))

        if reachable_edges:
            merged = unary_union(reachable_edges).buffer(BUFFER_M)
            isochrones.append(merged)
        else:
            isochrones.append(None)

    buildings['isochrone_15m'] = isochrones
    iso_gdf = buildings[['isochrone_15m']].copy()
    iso_gdf = iso_gdf.set_geometry('isochrone_15m')
    iso_gdf = iso_gdf.dropna(subset=['isochrone_15m'])

    print(f'Saving output to {OUTPUT_FILE.name}...')
    iso_gdf.to_file(OUTPUT_FILE, driver='GeoJSON')
    print('Done.')


if __name__ == '__main__':
    main() 