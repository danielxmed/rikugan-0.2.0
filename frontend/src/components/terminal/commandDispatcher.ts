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
