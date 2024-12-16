export interface IChatMessage {
    text: string;
    isMe?: boolean;
}

export interface IChatBotProps {
    closeChat: () => void;
    isLeft?: boolean;
}

export interface IChatWidgetProps {
    isLeft?: boolean;
}