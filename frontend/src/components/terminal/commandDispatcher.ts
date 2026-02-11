import type { ParsedCommand } from '../../types/commands';
import { wsService } from '../../services/websocket';
import { useViewportStore } from '../../stores/viewportStore';
import { loadModel, getModelInfo, showHelp } from './commands';

export async function dispatchCommand(
  command: ParsedCommand,
  writeLine: (text: string) => void,
): Promise<void> {
  switch (command.kind) {
    case 'load':
      await loadModel(command.modelName, writeLine);
      break;

    case 'run':
      wsService.send({
        type: 'inference.run',
        payload: { prompt: command.prompt },
      });
      break;

    case 'info':
      await getModelInfo(writeLine);
      break;

    case 'help':
      showHelp(writeLine);
      break;

    case 'contrast':
      useViewportStore.getState().setGamma(command.gamma);
      writeLine(`Contrast \u03B3 set to ${command.gamma}`);
      break;

    case 'slice':
      useViewportStore.getState().setSliceDepth(command.depth);
      if (command.depth === null) {
        writeLine('Slice plane \x1b[1moff\x1b[0m');
      } else {
        writeLine(`Slice depth set to \x1b[1m${command.depth.toFixed(2)}\x1b[0m`);
      }
      break;

    case 'layout': {
      if (command.iso) {
        useViewportStore.getState().setIsoLayout(command.mode, command.param);
      } else {
        useViewportStore.getState().setLayout(command.mode, command.param);
      }
      const label = command.iso ? 'Isolated layout' : 'Layout';
      if (command.param !== undefined) {
        writeLine(`${label} set to \x1b[1m${command.mode}\x1b[0m (${command.param})`);
      } else {
        writeLine(`${label} set to \x1b[1m${command.mode}\x1b[0m`);
      }
      break;
    }

    case 'get': {
      const vs = useViewportStore.getState();
      if (vs.isolatedLayers.length >= 4) {
        writeLine('\x1b[33mMax 4 isolated layers. Release one first.\x1b[0m');
      } else if (vs.isolatedLayers.includes(command.layer)) {
        writeLine(`\x1b[33mLayer ${command.layer} already isolated\x1b[0m`);
      } else {
        vs.isolateLayer(command.layer);
        writeLine(`Isolated \x1b[1mL${command.layer}\x1b[0m`);
      }
      break;
    }

    case 'release':
      useViewportStore.getState().releaseLayer(command.layer);
      if (command.layer === null) {
        writeLine('Released all isolated layers');
      } else {
        writeLine(`Released \x1b[1mL${command.layer}\x1b[0m`);
      }
      break;

    case 'error':
      writeLine(`\x1b[31m${command.message}\x1b[0m`);
      break;

    case 'unknown':
      writeLine(`\x1b[31mUnknown command: ${command.raw}\x1b[0m`);
      writeLine('Type \x1b[1mhelp\x1b[0m for a list of commands.');
      break;

    case 'empty':
      break;
  }
}
