import pickle
from typing import List, Tuple, Dict
import os

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes

from smirk.src.renderer.util import face_vertices, batch_orth_proj

def keep_vertices_and_update_faces(faces: torch.Tensor, uvfaces: torch.Tensor, vertices_to_keep: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep specified vertices in the mesh and update the faces.

    Args:
        faces (torch.Tensor): Tensor of shape (F, 3) representing faces, with each value being a vertex index.
        uvfaces (torch.Tensor): Tensor of shape (F, 3) representing UV faces, with each value being a vertex index.
        vertices_to_keep (List[int]): List or tensor of vertex indices to keep.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated vertices and faces tensors and uv faces tensor.
    """
    # Convert vertices_to_keep to a tensor if it's a list or numpy array
    if isinstance(vertices_to_keep, list) or isinstance(vertices_to_keep, np.ndarray):
        vertices_to_keep = torch.tensor(vertices_to_keep, dtype=torch.long)

    # Ensure vertices_to_keep is unique and sorted
    vertices_to_keep = torch.unique(vertices_to_keep)

    max_vertex_index = faces.max().long().item() + 1

    # ! not one-to-one correspondence between vertices and uv coords. but face and uvfaces are one-to-one correspondence
    # print("number of vertices, uvcoords:", max_vertex_index, uvfaces.max().long().item() + 1)

    # Create a mask for vertices to keep
    mask = torch.zeros(max_vertex_index, dtype=torch.bool)
    mask[vertices_to_keep] = True


    # Create a mapping from old vertex indices to new ones
    new_vertex_indices = torch.full((max_vertex_index,), -1, dtype=torch.long)
    new_vertex_indices[mask] = torch.arange(len(vertices_to_keep))

    # Remove faces that reference removed vertices (where mapping is -1)
    valid_faces_mask = (new_vertex_indices[faces] != -1).all(dim=1)
    filtered_faces = faces[valid_faces_mask]

    # Remove uv faces based on faces
    filtered_uvfaces = uvfaces[valid_faces_mask]

    # Update face indices to new vertex indices
    updated_faces = new_vertex_indices[filtered_faces]

    return updated_faces, filtered_uvfaces

class Renderer(nn.Module):
    """
    Renderer for FLAME head model using PyTorch3D.

    Attributes:
        image_size (int): Size of the rendered image.
        render_full_head (bool): Whether to render the full head or just the face.
        final_mask (List[int]): Mask for the final vertices to keep if not rendering full head.
    """
    def __init__(self, smirk_path:str = '', render_full_head:bool=False) -> None:
        """
        Initializes the Renderer.

        Args:
            smirk_path (str): Path to the SMIRK assets directory.
            render_full_head (bool): Whether to render the full head or just the face.
        """
        super(Renderer, self).__init__()
        self.image_size = 224
        self.render_full_head = render_full_head

        head_template_path = os.path.join(smirk_path, 'assets/head_template.obj')
        verts, faces, aux = load_obj(head_template_path)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        faces = faces.verts_idx[None,...]

        flame_mask_path = os.path.join(smirk_path, 'assets/FLAME_masks/FLAME_masks.pkl')
        flame_masks = pickle.load(open(flame_mask_path, 'rb'), encoding='latin1')

        if not render_full_head:
            self.final_mask = flame_masks['face'].tolist()

            # keep only faces that include vertices in face_mask
            faces, filtered_uvfaces = keep_vertices_and_update_faces(faces[0], uvfaces[0], self.final_mask)
            faces = faces.unsqueeze(0)
            uvfaces = filtered_uvfaces.unsqueeze(0)

        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)
        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)

    def forward(self, vertices: torch.Tensor, cam_params: torch.Tensor, **landmarks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the renderer.

        Args:
            vertices (torch.Tensor): Tensor of shape (B, V, 3) representing the vertices of the mesh.
            cam_params (torch.Tensor): Camera parameters for orthographic projection. [B, 3]
            landmarks (Dict[str, torch.Tensor]): {'landmarks_fan': [B, 68,3], 'landmarks_mp': [B, 105, 3], ...}

        Returns:
            Dict[str, torch.Tensor]: 
                - 'uv_grid': Rendered UV grid of shape (B, H, W, 2).
                - 'visibility_mask': Visibility mask of shape (B, H, W, 1).
        """
        transformed_vertices = batch_orth_proj(vertices, cam_params)
        transformed_vertices[:, :, 1:] = -transformed_vertices[:, :, 1:]

        transformed_landmarks = {}
        for key in landmarks.keys():
            transformed_landmarks[key] = batch_orth_proj(landmarks[key], cam_params)
            transformed_landmarks[key][:, :, 1:] = - transformed_landmarks[key][:, :, 1:]
            transformed_landmarks[key] = transformed_landmarks[key][...,:2]

        uv_grid, visibility_mask = self.render(vertices, transformed_vertices)

        outputs = {
            'uv_grid': uv_grid, 
            'visibility_mask': visibility_mask
        }

        return outputs
        
    def render(self, vertices: torch.Tensor, transformed_vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Render the mesh with the given vertices. Transformed vertices includes vertices in NDC space.
        Note that due to this custom implementation of the renderer, the NDC space does not follow the PyTorch3D convention of axes.

        Args:
            vertices (torch.Tensor): Tensor of shape (B, V, 3) representing the vertices of the mesh.
            transformed_vertices (torch.Tensor): Transformed vertices after applying camera projection.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rendered UV grid and visibility mask.
        """
        batch_size = vertices.shape[0]
        
        if not self.render_full_head:
            transformed_vertices = transformed_vertices[:,self.final_mask,:]
            vertices = vertices[:,self.final_mask,:]

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        
        # attributes
        uv_coords = self.face_uvcoords.expand(batch_size, -1, -1, -1)

        attributes = uv_coords
        # rasterize
        rendering = self.rasterize(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        
        uv_grid = rendering.permute(0, 2, 3, 1).contiguous()  
        
        return uv_grid[:, :,:,0:2], uv_grid[:, :, :, 3:4]  # return uv grid and visibility mask

    def rasterize(self, vertices: torch.Tensor, faces: torch.Tensor, attributes: torch.Tensor, h: int=None, w: int=None) -> torch.Tensor:
        """
        Rasterize the mesh with the given vertices and faces.

        Args:
            vertices (torch.Tensor): Tensor of shape (B, V, 3) representing the vertices of the mesh.
            faces (torch.Tensor): Tensor of shape (B, F, 3) representing the faces of the mesh.
            attributes (torch.Tensor): Tensor of shape (B, F, 3, D) representing the attributes for each face.
            h (int, optional): Height of the output image. Defaults to None.
            w (int, optional): Width of the output image. Defaults to None.

        Returns:
            torch.Tensor: Rendered pixel values of shape (B, H, W, D), where D is the number of attributes.
        """
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]

        if h is None and w is None:
            image_size = self.image_size
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h

        # MPS does not support the cumsum operation on int64 data, which is a must in the rasterize_meshes function. Therefore, we must move it to cpu.
        orig_device = fixed_vertices.device.type
        if orig_device == 'mps':
            meshes_screen = Meshes(verts=fixed_vertices.float().cpu(), faces=faces.long().cpu())
        else:
            meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.to(fixed_vertices.device).long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=False,
        )
        if orig_device == 'mps':
            pix_to_face = pix_to_face.to(torch.device(orig_device))
            bary_coords = bary_coords.to(torch.device(orig_device))
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals
