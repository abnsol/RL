"""
Pygame-based visual renderer for The Last Signal.
Provides optional graphical visualization of the game state.
"""

import pygame
import numpy as np
from typing import Optional, Tuple

from game_engine import GameEngine
from game_config import GameConfig


class GameRenderer:
    """Renders The Last Signal game state using Pygame."""
    
    def __init__(self, game: GameEngine, cell_size: int = 50, fps: int = 2, fullscreen: bool = False):
        """
        Initialize the renderer.
        
        Args:
            game: GameEngine instance
            cell_size: Size of each grid cell in pixels
            fps: Frames per second for rendering
        """
        self.game = game
        self.cell_size = cell_size
        self.fps = fps
        self.fullscreen = fullscreen

        # If fullscreen requested, compute cell_size to fit the display
        if self.fullscreen:
            # Initialize pygame so display info is available
            pygame.init()
            info = pygame.display.Info()
            display_w, display_h = info.current_w, info.current_h

            # Reserve space for info panel and margins
            panel_reserved = 500
            vert_margin = 10

            usable_w = max(30, display_w - panel_reserved)
            usable_h = max(30, display_h - vert_margin)

            # Compute max cell size that fits the grid in the usable area
            max_cell_w = usable_w // game.config.grid_width
            max_cell_h = usable_h // game.config.grid_height
            computed = max(4, min(max_cell_w, max_cell_h))
            self.cell_size = computed

            # Set final window size to full display
            self.width = display_w
            self.height = display_h
        else:
            # Calculate window size for windowed mode
            self.width = game.config.grid_width * cell_size + 200
            self.height = game.config.grid_height * cell_size + 100
        
        # Colors
        self.COLOR_BACKGROUND = (20, 20, 30)
        self.COLOR_EMPTY = (40, 40, 60)
        self.COLOR_AGENT = (100, 200, 100)
        self.COLOR_SIGNAL = (255, 215, 0)
        self.COLOR_LOW_HAZARD = (100, 150, 255)
        self.COLOR_MED_HAZARD = (255, 150, 50)
        self.COLOR_HIGH_HAZARD = (255, 50, 50)
        self.COLOR_TEXT = (200, 200, 200)
        self.COLOR_VISITED = (60, 80, 100)
        
        # Initialize Pygame and window
        if not self.fullscreen:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            # Already initialized above when computing sizes
            # Create fullscreen surface
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("The Last Signal")
        self.clock = pygame.time.Clock()
        # Scale fonts relative to cell size for readability on large screens
        base_font_size = max(12, self.cell_size // 2)
        self.font = pygame.font.Font(None, base_font_size + 6)
        self.font_small = pygame.font.Font(None, base_font_size)
        
        self.running = True
        # Pause/step controls for debugging
        self.paused = False
        self.step = False

    def render(self, show_info: bool = True):
        """
        Render one frame of the game.
        
        Args:
            show_info: Whether to display info panel
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_p:
                    # Toggle pause/resume
                    self.paused = not self.paused
                    # Clear any single-step request when toggling
                    self.step = False
                elif event.key == pygame.K_n:
                    # Single-step when paused
                    self.step = True
        
        # Clear screen
        self.screen.fill(self.COLOR_BACKGROUND)
        
        # Draw grid
        self._draw_grid()
        
        # Draw cells
        self._draw_cells()
        
        # Draw agent
        self._draw_agent()
        
        # Draw info panel
        if show_info:
            self._draw_info_panel()

        # If paused, draw an overlay and message
        if self.paused:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 4))
            self.screen.blit(overlay, (0, 0))

            # Large paused text
            large_size = max(24, self.cell_size)
            try:
                large_font = pygame.font.Font(None, large_size * 2)
            except Exception:
                large_font = self.font

            paused_surf = large_font.render("PAUSED", True, (255, 255, 255))
            paused_rect = paused_surf.get_rect(center=(self.width // 2, self.height // 2 - 20))
            self.screen.blit(paused_surf, paused_rect)

            hint_surf = self.font_small.render("P = Resume / N = Step", True, (220, 220, 220))
            hint_rect = hint_surf.get_rect(center=(self.width // 2, self.height // 2 + 20))
            self.screen.blit(hint_surf, hint_rect)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_grid(self):
        """Draw grid lines."""
        grid_color = (50, 50, 70)
        
        # Vertical lines
        for x in range(self.game.config.grid_width + 1):
            pygame.draw.line(
                self.screen, grid_color,
                (x * self.cell_size, 0),
                (x * self.cell_size, self.game.config.grid_height * self.cell_size),
                1
            )
        
        # Horizontal lines
        for y in range(self.game.config.grid_height + 1):
            pygame.draw.line(
                self.screen, grid_color,
                (0, y * self.cell_size),
                (self.game.config.grid_width * self.cell_size, y * self.cell_size),
                1
            )
    
    def _draw_cells(self):
        """Draw all cells with their contents."""
        for y in range(self.game.config.grid_height):
            for x in range(self.game.config.grid_width):
                cell = self.game.grid[y][x]
                rect = pygame.Rect(
                    x * self.cell_size, y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                # Draw cell background based on hazard level
                hazard_prob = cell.hazard_probability
                if (x, y) in self.game.visited_cells:
                    color = self.COLOR_VISITED
                else:
                    color = self.COLOR_EMPTY
                
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw hazard intensity
                if hazard_prob > 0:
                    if hazard_prob > 0.6:
                        hazard_color = self.COLOR_HIGH_HAZARD
                    elif hazard_prob > 0.3:
                        hazard_color = self.COLOR_MED_HAZARD
                    else:
                        hazard_color = self.COLOR_LOW_HAZARD
                    
                    # Draw hazard indicator (small corner square)
                    hazard_size = max(5, int((hazard_prob * 0.8) * self.cell_size))
                    pygame.draw.rect(
                        self.screen, hazard_color,
                        pygame.Rect(rect.right - hazard_size, rect.top,
                                  hazard_size, hazard_size)
                    )
                
                # Draw signal nodes
                if cell.has_signal:
                    if not cell.signal_collected:
                        pygame.draw.circle(
                            self.screen, self.COLOR_SIGNAL,
                            (rect.centerx, rect.centery),
                            self.cell_size // 4
                        )
                    else:
                        pygame.draw.circle(
                            self.screen, (150, 150, 150),
                            (rect.centerx, rect.centery),
                            self.cell_size // 4
                        )
    
    def _draw_agent(self):
        """Draw the agent."""
        rect = pygame.Rect(
            self.game.agent_x * self.cell_size,
            self.game.agent_y * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.circle(
            self.screen, self.COLOR_AGENT,
            (rect.centerx, rect.centery),
            self.cell_size // 3
        )
    
    def _draw_info_panel(self):
        """Draw information panel on the right side."""
        panel_x = self.game.config.grid_width * self.cell_size + 10
        panel_y = 10
        
        info_texts = [
            f"Step: {self.game.episode_step}",
            f"Health: {self.game.health}/{self.game.config.max_health}",
            f"Time: {self.game.time_remaining}/{self.game.config.time_budget}",
            f"Position: ({self.game.agent_x}, {self.game.agent_y})",
            f"Signals: {sum(1 for y in range(self.game.config.grid_height) for x in range(self.game.config.grid_width) if self.game.grid[y][x].signal_collected)}/{self.game.config.num_signals}",
            f"Visited: {len(self.game.visited_cells)} cells",
            "",
            "Actions:",
            "↑/↓/←/→ = Move",
            "S = Stabilize",
            "W = Wait",
            "ESC = Exit",
        ]
        
        for i, text in enumerate(info_texts):
            surf = self.font_small.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surf, (panel_x, panel_y + i * 20))
    
    def close(self):
        """Close the renderer."""
        pygame.quit()
        self.running = False


def interactive_play(config: GameConfig = None, fullscreen: bool = True):
    """
    Run an interactive game session with keyboard control.
    Useful for understanding the environment dynamics.
    
    Args:
        config: GameConfig object. Uses defaults if None.
        fullscreen: If True, scale the renderer to fullscreen.
    """
    if config is None:
        config = GameConfig()

    game = GameEngine(config)
    renderer = GameRenderer(game, fullscreen=fullscreen)
    
    action_map = {
        pygame.K_UP: 0,      # MOVE_UP
        pygame.K_DOWN: 1,    # MOVE_DOWN
        pygame.K_LEFT: 2,    # MOVE_LEFT
        pygame.K_RIGHT: 3,   # MOVE_RIGHT
        pygame.K_s: 4,       # STABILIZE
        pygame.K_w: 5,       # WAIT
    }
    
    game.reset()
    total_score = 0.0
    
    while renderer.running:
        # If paused and no single-step requested, just render and continue
        if renderer.paused and not renderer.step:
            renderer.render()
            continue

        # Handle keyboard input
        keys = pygame.key.get_pressed()
        action = None

        for key, act in action_map.items():
            if keys[key]:
                action = act
                break

        if action is None:
            action = 5  # Default to WAIT

        # Execute step (normal or single-step while paused)
        obs, reward, terminated, info = game.step(action)
        # Accumulate scalar reward (RewardVector has `total` property)
        try:
            total_score += float(reward.total)
        except Exception:
            # Fallback if reward doesn't have total
            total_score += 0.0

        # Clear single-step flag after consuming it
        renderer.step = False

        # Render
        renderer.render()
        
        if terminated:
            print(f"\nEpisode ended!")
            print(f"Final health: {game.health}")
            print(f"Final score: {total_score:.2f}")
            print(f"Signals collected: {info.get('signals_collected', 'N/A')}")
            break
    
    renderer.close()
